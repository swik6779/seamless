# Seamless image processing

## Build

```bash
make seamless
make test_pyramid
./test_pyramid
```

The CUDA target is optional:

```bash
make seamless_cuda
```

## Commands

Gaussian blur:

```bash
./seamless gauss-blur --input input.png --output blurred.png \
  --backend cpu --kernel-size 7
```

On a CUDA-capable build, use `--backend cuda` for Gaussian blur, resize, or
Laplacian blending. CUDA commands report host-to-device transfer, kernel, and
device-to-host times separately.

Resize while preserving aspect ratio. Specify exactly one target dimension:

```bash
./seamless resize --input input.png --output resized.png \
  --width 640 --backend omp
```

Laplacian pyramid blending:

```bash
./seamless laplac-blend --a apple.png --b orange.png \
  --mask mask.png --output blended.png --backend cpu \
  --kernel-size 7 --levels 6
```

CUDA examples:

```bash
./seamless resize --input input.png --output resized.png \
  --height 720 --backend cuda

./seamless laplac-blend --a apple.png --b orange.png \
  --mask mask.png --output blended.png --backend cuda \
  --kernel-size 7 --levels 6
```

Image A defines the blend dimensions. If image B or the mask has different
spatial dimensions, it is automatically resized to image A using bilinear
interpolation. Image A and B must still have the same number of channels.

Internally, pyramid operations use float32 data and clamp-to-edge borders.
Conversion back to 8-bit occurs only when writing the final image.

## Blend benchmark

Run serial CPU, OpenMP thread-count variants, and CUDA when
`./seamless_cuda` exists:

```bash
scripts/benchmark_blend.sh \
  --a apple.png --b orange.png --mask mask.png \
  --levels 3,4,5,6 \
  --kernels 3,5,7,9,11 \
  --threads 1,2,4,8 \
  --repetitions 3 \
  --output benchmark_blend.csv
```

The CSV contains CPU/OpenMP total and per-stage timings. CUDA rows contain
the transfer-inclusive derived time plus separate host-to-device, kernel, and
device-to-host timings.
