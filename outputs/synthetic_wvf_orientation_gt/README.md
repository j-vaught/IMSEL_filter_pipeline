# Synthetic WVF orientation ground-truth comparison

This run compares closed-form steerable WVF orientation against brute sampled directional-search orientation and analytic synthetic-shape ground truth.

`steerable_atan2` uses `atan2(Gy, Gx)`. `brute_N` quantizes the same steerable direction onto `N` sampled orientations in `[0, pi)`, which is equivalent to rotating the isotropic WVF directional filter bank after the direct-vs-steered kernel equivalence has been established.

Primary metrics use the `smooth` boundary mask, which excludes polygon vertices and square corners by `vertex_exclude_px` because corner orientation is not single-valued.
