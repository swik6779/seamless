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

Image A defines the blend dimensions. If image B or the mask has different
spatial dimensions, it is automatically resized to image A using bilinear
interpolation. Image A and B must still have the same number of channels.

Internally, pyramid operations use float32 data and clamp-to-edge borders.
Conversion back to 8-bit occurs only when writing the final image.
