#!/usr/bin/env python3

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HOME = Path.home()
PROJECT = HOME / "Desktop" / "benchmarking" / "seamless"

DEFAULT_A = HOME / "Desktop" / "benchmarking" / "test" / "pillars.jpg"
DEFAULT_B = HOME / "Desktop" / "benchmarking" / "test" / "rose.jpg"
DEFAULT_MASK = HOME / "Desktop" / "benchmarking" / "test" / "mask_grey.jpeg"
DEFAULT_CPU = PROJECT / "seamless"
DEFAULT_CUDA = PROJECT / "seamless_cuda"
DEFAULT_OUTPUT = HOME / "Desktop" / "benchmarking" / "benchmark_blend.csv"

CSV_COLUMNS = [
    "backend",
    "threads",
    "levels",
    "kernel_size",
    "repeat",
    "total_ms",
    "blur_ms",
    "downsample_ms",
    "expand_ms",
    "subtract_ms",
    "blend_ms",
    "collapse_ms",
    "h2d_ms",
    "cuda_kernel_ms",
    "d2h_ms",
    "throughput_mpix_s",
    "speedup_vs_serial",
    "speedup_kernel_vs_serial",
    "speedup_vs_openmp",
]

SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF,
}


def parse_csv_ints(value):
    try:
        values = [int(item.strip()) for item in value.split(",")]
        if not values or any(item <= 0 for item in values):
            raise ValueError
        return values
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Use comma-separated positive integers, e.g. 1,2,4"
        )


def get_jpeg_dimensions(image_path):
    """Read JPEG width and height without external packages."""
    try:
        with open(image_path, "rb") as image:
            if image.read(2) != b"\xff\xd8":
                raise RuntimeError(
                    f"Input is not a JPEG file: {image_path}"
                )

            while True:
                byte = image.read(1)

                if not byte:
                    break

                if byte != b"\xff":
                    continue

                marker_byte = image.read(1)

                while marker_byte == b"\xff":
                    marker_byte = image.read(1)

                if not marker_byte:
                    break

                marker = marker_byte[0]

                # Standalone markers do not have segment lengths.
                if (
                    marker == 0x01
                    or marker in (0xD8, 0xD9)
                    or 0xD0 <= marker <= 0xD7
                ):
                    continue

                length_data = image.read(2)

                if len(length_data) != 2:
                    break

                segment_length = int.from_bytes(length_data, "big")

                if segment_length < 2:
                    break

                if marker in SOF_MARKERS:
                    data = image.read(5)

                    if len(data) != 5:
                        break

                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")

                    if width > 0 and height > 0:
                        return width, height

                    break

                image.seek(segment_length - 2, os.SEEK_CUR)

    except OSError as exc:
        raise RuntimeError(
            f"Unable to read image {image_path}: {exc}"
        ) from exc

    raise RuntimeError(
        f"Could not determine JPEG dimensions: {image_path}"
    )


def extract_value(label, output, required=True):
    """
    Find a timing value in program output.

    Returns None when an optional timing is not present.
    """
    for line in output.splitlines():
        if re.search(label, line, re.IGNORECASE):
            matches = re.findall(
                r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?",
                line,
            )

            if matches:
                return float(matches[-1])

    if required:
        raise RuntimeError(
            f"Could not find timing '{label}' in program output.\n"
            f"Program output:\n{output}"
        )

    return None


def run_command(command, env=None):
    result = subprocess.run(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        print(
            "\nCommand failed:",
            " ".join(str(arg) for arg in command),
            file=sys.stderr,
        )
        print(result.stdout, file=sys.stderr)

        raise RuntimeError(
            f"Process exited with status {result.returncode}"
        )

    return result.stdout


def is_executable(path):
    return path.is_file() and os.access(path, os.X_OK)


def format_value(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def calculate_throughput(width, height, milliseconds):
    if milliseconds is None or milliseconds <= 0:
        return None

    megapixels = width * height / 1_000_000.0
    return megapixels / (milliseconds / 1000.0)


def calculate_speedup(reference, measured):
    if (
        reference is None
        or measured is None
        or reference <= 0
        or measured <= 0
    ):
        return None

    return reference / measured


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark serial CPU, OpenMP, and CUDA "
            "Laplacian blending."
        )
    )

    parser.add_argument("--a", default=str(DEFAULT_A), help="Image A")
    parser.add_argument("--b", default=str(DEFAULT_B), help="Image B")
    parser.add_argument("--mask", default=str(DEFAULT_MASK), help="Mask")
    parser.add_argument(
        "--levels",
        type=parse_csv_ints,
        default=[1, 2, 3, 4, 5, 6],
    )
    parser.add_argument(
        "--kernels",
        type=parse_csv_ints,
        default=[3, 5, 7, 9, 11],
    )
    parser.add_argument(
        "--threads",
        type=parse_csv_ints,
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--cpu-exe", default=str(DEFAULT_CPU))
    parser.add_argument("--cuda-exe", default=str(DEFAULT_CUDA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT)

    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be a positive integer")

    # Expand ~ and resolve input paths.
    args.a = str(Path(args.a).expanduser().resolve())
    args.b = str(Path(args.b).expanduser().resolve())
    args.mask = str(Path(args.mask).expanduser().resolve())

    for label, path in (
        ("Image A", args.a),
        ("Image B", args.b),
        ("Mask", args.mask),
    ):
        if not Path(path).is_file():
            parser.error(f"{label} does not exist: {path}")

    cpu_exe = Path(args.cpu_exe).expanduser().resolve()
    cuda_exe = Path(args.cuda_exe).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not is_executable(cpu_exe):
        parser.error(
            f"CPU/OpenMP executable is not executable: {cpu_exe}"
        )

    width, height = get_jpeg_dimensions(args.a)

    print(f"Image dimensions: {width} x {height}")

    have_cuda = is_executable(cuda_exe)

    if not have_cuda:
        print(
            f"CUDA executable unavailable; skipping CUDA: {cuda_exe}",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="benchmark_blend_"
    ) as temporary_directory:

        temp_dir = Path(temporary_directory)

        with output_path.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_COLUMNS,
            )
            writer.writeheader()
            csv_file.flush()

            def write_row(row):
                writer.writerow(row)
                csv_file.flush()

            def run_cpu_or_omp(
                backend,
                thread_count,
                level_count,
                kernel_size,
                repeat,
            ):
                output_image = temp_dir / (
                    f"{backend}_{thread_count}_{level_count}_"
                    f"{kernel_size}_{repeat}.png"
                )

                command = [
                    str(cpu_exe),
                    "laplac-blend",
                    "--a", args.a,
                    "--b", args.b,
                    "--mask", args.mask,
                    "--output", str(output_image),
                    "--backend", backend,
                    "--kernel-size", str(kernel_size),
                    "--levels", str(level_count),
                ]

                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = str(
                    thread_count if backend == "omp" else 1
                )

                program_output = run_command(command, env=env)

                # The original Bash script assumes this value is seconds.
                total_seconds = extract_value(
                    r"Elapsed time:", program_output
                )
                total_ms = total_seconds * 1000.0

                # Stage timings are optional. Missing values remain blank.
                stage_patterns = {
                    "blur_ms": r"Stage blur:",
                    "downsample_ms": r"Stage downsample:",
                    "expand_ms": r"Stage expand:",
                    "subtract_ms": r"Stage subtract:",
                    "blend_ms": r"Stage blend:",
                    "collapse_ms": r"Stage collapse:",
                }

                stages = {
                    column: extract_value(
                        pattern, program_output, required=False
                    )
                    for column, pattern in stage_patterns.items()
                }

                if backend == "cpu":
                    run_cpu_or_omp.serial_time_ms = total_ms
                    speedup_serial = 1.0
                else:
                    speedup_serial = calculate_speedup(
                        run_cpu_or_omp.serial_time_ms,
                        total_ms,
                    )

                row = {
                    "backend": backend,
                    "threads": thread_count,
                    "levels": level_count,
                    "kernel_size": kernel_size,
                    "repeat": repeat,
                    "total_ms": format_value(total_ms),
                    **{
                        column: format_value(value)
                        for column, value in stages.items()
                    },
                    "h2d_ms": "",
                    "cuda_kernel_ms": "",
                    "d2h_ms": "",
                    "throughput_mpix_s": format_value(
                        calculate_throughput(
                            width, height, total_ms
                        )
                    ),
                    "speedup_vs_serial": format_value(
                        speedup_serial
                    ),
                    "speedup_kernel_vs_serial": "",
                    "speedup_vs_openmp": "",
                }

                write_row(row)

                print(
                    f"{backend.upper():4} "
                    f"threads={thread_count} "
                    f"levels={level_count} "
                    f"kernel={kernel_size} "
                    f"repeat={repeat} "
                    f"total={total_ms:.3f} ms"
                )

                return total_ms

            def run_cuda(
                level_count,
                kernel_size,
                repeat,
                serial_time_ms,
                best_omp_time_ms,
            ):
                output_image = temp_dir / (
                    f"cuda_{level_count}_{kernel_size}_{repeat}.png"
                )

                command = [
                    str(cuda_exe),
                    "laplac-blend",
                    "--a", args.a,
                    "--b", args.b,
                    "--mask", args.mask,
                    "--output", str(output_image),
                    "--backend", "cuda",
                    "--kernel-size", str(kernel_size),
                    "--levels", str(level_count),
                ]

                program_output = run_command(command)

                # These CUDA timing labels are required for CUDA rows.
                h2d_ms = extract_value(
                    r"CUDA host-to-device:", program_output
                )
                kernel_ms = extract_value(
                    r"CUDA kernels:", program_output
                )
                d2h_ms = extract_value(
                    r"CUDA device-to-host:", program_output
                )

                # Preserve the Bash script's timing methodology.
                total_ms = h2d_ms + kernel_ms + d2h_ms

                row = {
                    "backend": "cuda",
                    "threads": "",
                    "levels": level_count,
                    "kernel_size": kernel_size,
                    "repeat": repeat,
                    "total_ms": format_value(total_ms),
                    "blur_ms": "",
                    "downsample_ms": "",
                    "expand_ms": "",
                    "subtract_ms": "",
                    "blend_ms": "",
                    "collapse_ms": "",
                    "h2d_ms": format_value(h2d_ms),
                    "cuda_kernel_ms": format_value(kernel_ms),
                    "d2h_ms": format_value(d2h_ms),
                    "throughput_mpix_s": format_value(
                        calculate_throughput(
                            width, height, total_ms
                        )
                    ),
                    "speedup_vs_serial": format_value(
                        calculate_speedup(serial_time_ms, total_ms)
                    ),
                    "speedup_kernel_vs_serial": format_value(
                        calculate_speedup(serial_time_ms, kernel_ms)
                    ),
                    "speedup_vs_openmp": format_value(
                        calculate_speedup(best_omp_time_ms, total_ms)
                    ),
                }

                write_row(row)

                print(
                    f"CUDA levels={level_count} "
                    f"kernel={kernel_size} "
                    f"repeat={repeat} "
                    f"total={total_ms:.3f} ms "
                    f"(H2D={h2d_ms:.3f}, "
                    f"kernel={kernel_ms:.3f}, "
                    f"D2H={d2h_ms:.3f})"
                )

            for level_count in args.levels:
                for kernel_size in args.kernels:
                    for repeat in range(
                        1, args.repetitions + 1
                    ):
                        run_cpu_or_omp.serial_time_ms = None

                        serial_time_ms = run_cpu_or_omp(
                            "cpu",
                            1,
                            level_count,
                            kernel_size,
                            repeat,
                        )

                        omp_times = []

                        for thread_count in args.threads:
                            omp_time = run_cpu_or_omp(
                                "omp",
                                thread_count,
                                level_count,
                                kernel_size,
                                repeat,
                            )
                            omp_times.append(omp_time)

                        best_omp_time_ms = (
                            min(omp_times) if omp_times else None
                        )

                        if have_cuda:
                            run_cuda(
                                level_count,
                                kernel_size,
                                repeat,
                                serial_time_ms,
                                best_omp_time_ms,
                            )

    print(f"\nBenchmark results written to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
