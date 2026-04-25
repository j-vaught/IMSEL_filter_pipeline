# Synthetic WVF steerability check

This run tests whether a directional WVF response at 23.5 degrees is equivalent to steering the two canonical WVF derivative components.

For each image, order, and radius, the script computes

```text
G_theta = cos(theta) * Gx + sin(theta) * Gy
```

and compares it with a direct WVF convolution whose least-squares Taylor target is the directional derivative

```text
partial_theta f = cos(theta) * f_x + sin(theta) * f_y.
```

The disk support is isotropic and fixed at radius 25 pixels. The test covers all 21 nested-shape synthetic images at 1024, 2048, and 4096 pixels, for Taylor orders d=2 and d=4.

## Result

The directional WVF is steerable from `Gx` and `Gy` to float32 precision in this implementation.

The direct directional kernel and the steered kernel agree to about `1e-20` in coefficient space. The full-image response comparisons agree to about `1e-7` absolute error, with minimum response correlation above `0.9999999997`. The largest relative errors occur on low-contrast palettes because the response magnitude is intentionally small, not because the absolute error increases.

## Files

- `summary.json` gives the aggregate result.
- `metrics.csv` and `metrics.json` give per-image metrics.
- `figures/` contains contact sheets for d=2 and d=4 at each image size.
