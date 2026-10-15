#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/benchmark_blend.sh \
    --a IMAGE_A --b IMAGE_B --mask MASK \
    [--levels 3,5,6] [--kernels 3,5,7,9,11] \
    [--threads 1,2,4,8] [--repetitions N] \
    [--cpu-exe ./seamless] [--cuda-exe ./seamless_cuda] \
    [--output benchmark.csv]

The input images must already be suitable for laplac-blend. Image A's
dimensions are detected automatically and used for throughput calculation.
EOF
}

image_a=""
image_b=""
mask=""
levels_csv="1,2,3,4,5,6"
kernels_csv="3,5,7,9,11"
threads_csv="1,2,4,8"
repetitions=1
cpu_exe="./seamless"
cuda_exe="./seamless_cuda"
output_csv="benchmark_blend.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --a) image_a="$2"; shift 2 ;;
    --b) image_b="$2"; shift 2 ;;
    --mask) mask="$2"; shift 2 ;;
    --levels) levels_csv="$2"; shift 2 ;;
    --kernels) kernels_csv="$2"; shift 2 ;;
    --threads) threads_csv="$2"; shift 2 ;;
    --repetitions) repetitions="$2"; shift 2 ;;
    --cpu-exe) cpu_exe="$2"; shift 2 ;;
    --cuda-exe) cuda_exe="$2"; shift 2 ;;
    --output) output_csv="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$image_a" || -z "$image_b" || -z "$mask" ]]; then
  echo "--a, --b, and --mask are required" >&2
  usage >&2
  exit 2
fi

if ! [[ "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
  echo "repetitions must be a positive integer" >&2
  exit 2
fi

if [[ ! -x "$cpu_exe" ]]; then
  echo "CPU/OpenMP executable is not executable: $cpu_exe" >&2
  exit 2
fi

image_info="$($cpu_exe image-info --input "$image_a")"
width="$(awk -F= '$1 == "width" { print $2 }' <<< "$image_info")"
height="$(awk -F= '$1 == "height" { print $2 }' <<< "$image_info")"
if ! [[ "$width" =~ ^[1-9][0-9]*$ && "$height" =~ ^[1-9][0-9]*$ ]]; then
  echo "could not determine image A dimensions" >&2
  exit 1
fi

if [[ ! -x "$cuda_exe" ]]; then
  echo "CUDA executable not found; CUDA rows will be skipped: $cuda_exe" >&2
  have_cuda=0
else
  have_cuda=1
fi

IFS=',' read -r -a levels <<< "$levels_csv"
IFS=',' read -r -a kernels <<< "$kernels_csv"
IFS=',' read -r -a threads <<< "$threads_csv"

temp_dir="$(mktemp -d)"
trap 'rm -rf "$temp_dir"' EXIT

echo "backend,threads,levels,kernel_size,repeat,total_ms,blur_ms,downsample_ms,expand_ms,subtract_ms,blend_ms,collapse_ms,h2d_ms,cuda_kernel_ms,d2h_ms,throughput_mpix_s,speedup_vs_serial,speedup_kernel_vs_serial,speedup_vs_openmp" > "$output_csv"

declare -A omp_times_ms
serial_time_ms=""
best_omp_time_ms=""

extract_value() {
  local label="$1"
  local text="$2"
  awk -v label="$label" '$0 ~ label {gsub(/[^0-9.]/, "", $NF); print $NF; exit}' <<< "$text"
}

run_cpu_or_omp() {
  local backend="$1"
  local omp_threads="$2"
  local level_count="$3"
  local kernel_size="$4"
  local repeat="$5"
  local output_path="$temp_dir/${backend}_${omp_threads}_${level_count}_${kernel_size}_${repeat}.png"
  local text
  if [[ "$backend" == "omp" ]]; then
    text="$(OMP_NUM_THREADS="$omp_threads" "$cpu_exe" laplac-blend \
      --a "$image_a" --b "$image_b" --mask "$mask" --output "$output_path" \
      --backend omp --kernel-size "$kernel_size" --levels "$level_count" 2>&1)"
  else
    text="$(OMP_NUM_THREADS=1 "$cpu_exe" laplac-blend \
      --a "$image_a" --b "$image_b" --mask "$mask" --output "$output_path" \
      --backend cpu --kernel-size "$kernel_size" --levels "$level_count" 2>&1)"
  fi
  local total_s blur downsample expand subtract blend collapse
  total_s="$(extract_value 'Elapsed time:' "$text")"
  blur="$(extract_value 'Stage blur:' "$text")"
  downsample="$(extract_value 'Stage downsample:' "$text")"
  expand="$(extract_value 'Stage expand:' "$text")"
  subtract="$(extract_value 'Stage subtract:' "$text")"
  blend="$(extract_value 'Stage blend:' "$text")"
  collapse="$(extract_value 'Stage collapse:' "$text")"
  local total_ms throughput
  total_ms="$(awk -v seconds="$total_s" 'BEGIN { printf "%.6f", seconds * 1000.0 }')"
  throughput="$(awk -v w="$width" -v h="$height" -v ms="$total_ms" 'BEGIN { printf "%.6f", (w*h/1000000.0)/(ms/1000.0) }')"
  local speedup_vs_serial=""
  if [[ "$backend" == "cpu" ]]; then
    serial_time_ms="$total_ms"
    speedup_vs_serial="1.000000"
  else
    omp_times_ms["$omp_threads"]="$total_ms"
    speedup_vs_serial="$(awk -v serial="$serial_time_ms" -v current="$total_ms" 'BEGIN { printf "%.6f", serial/current }')"
    if [[ -z "$best_omp_time_ms" ]] || awk -v current="$total_ms" -v best="$best_omp_time_ms" 'BEGIN { exit !(current < best) }'; then
      best_omp_time_ms="$total_ms"
    fi
  fi
  echo "$backend,$omp_threads,$level_count,$kernel_size,$repeat,$total_ms,$blur,$downsample,$expand,$subtract,$blend,$collapse,,,,,$throughput,$speedup_vs_serial,," >> "$output_csv"
}

run_cuda() {
  local level_count="$1"
  local kernel_size="$2"
  local repeat="$3"
  local output_path="$temp_dir/cuda_${level_count}_${kernel_size}_${repeat}.png"
  local text
  text="$("$cuda_exe" laplac-blend \
    --a "$image_a" --b "$image_b" --mask "$mask" --output "$output_path" \
    --backend cuda --kernel-size "$kernel_size" --levels "$level_count" 2>&1)"
  local h2d kernel d2h throughput
  h2d="$(extract_value 'CUDA host-to-device:' "$text")"
  kernel="$(extract_value 'CUDA kernels:' "$text")"
  d2h="$(extract_value 'CUDA device-to-host:' "$text")"
  local with_transfers
  with_transfers="$(awk -v a="$h2d" -v b="$kernel" -v c="$d2h" 'BEGIN { printf "%.6f", a+b+c }')"
  throughput="$(awk -v w="$width" -v h="$height" -v ms="$with_transfers" 'BEGIN { printf "%.6f", (w*h/1000000.0)/(ms/1000.0) }')"
  local speedup_with_transfers speedup_kernel speedup_openmp
  speedup_with_transfers="$(awk -v serial="$serial_time_ms" -v cuda="$with_transfers" 'BEGIN { printf "%.6f", serial/cuda }')"
  speedup_kernel="$(awk -v serial="$serial_time_ms" -v cuda="$kernel" 'BEGIN { printf "%.6f", serial/cuda }')"
  speedup_openmp="$(awk -v omp="$best_omp_time_ms" -v cuda="$with_transfers" 'BEGIN { printf "%.6f", omp/cuda }')"
  echo "cuda,,${level_count},${kernel_size},${repeat},${with_transfers},,,,,,$h2d,$kernel,$d2h,$throughput,$speedup_with_transfers,$speedup_kernel,$speedup_openmp" >> "$output_csv"
}

for level_count in "${levels[@]}"; do
  for kernel_size in "${kernels[@]}"; do
    for repeat in $(seq 1 "$repetitions"); do
      serial_time_ms=""
      best_omp_time_ms=""
      run_cpu_or_omp cpu 1 "$level_count" "$kernel_size" "$repeat"
      for omp_threads in "${threads[@]}"; do
        run_cpu_or_omp omp "$omp_threads" "$level_count" "$kernel_size" "$repeat"
      done
      if [[ "$have_cuda" -eq 1 ]]; then
        run_cuda "$level_count" "$kernel_size" "$repeat"
      fi
    done
  done
done

echo "Benchmark results written to $output_csv"
