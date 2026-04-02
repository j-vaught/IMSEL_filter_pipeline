#set document(title: "Equivalent Forms of the Fused Stencil Filter")
#set page(margin: 1in, numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

= Equivalent Forms of the Fused Stencil Filter

The fused stencil filter can be expressed in several equivalent notations. Each form emphasizes a different aspect of the computation and connects to different implementation strategies and bodies of literature.

== Summation Form

The original per-pixel formulation iterates over all unique stencil positions for a given orientation.

$ R_k = sum_(ell=1)^(N'_k) alpha_(k, ell) dot f(X_0 + tilde(delta)_ell^x, space Y_0 + tilde(delta)_ell^y), quad k^* = arg max_k |R_k| $

This form makes the structure of the computation explicit. Each $alpha_(k,ell)$ is a precomputed scalar weight encoding polynomial fitting, pseudoinverse extraction, and Gaussian line weighting. The summation runs over $N'_k$ unique pixel positions in the deduplicated stencil for orientation $k$.

== Inner Product Form

Collecting the weights and sampled intensities into vectors yields a compact algebraic expression.

$ R_k = chevron.l bold(alpha)_k, space bold(f)_k chevron.r $

Here $bold(alpha)_k in bb(R)^(N'_k)$ is the vector of fused weights for orientation $k$, and $bold(f)_k in bb(R)^(N'_k)$ is the vector of image intensities sampled at the corresponding stencil positions. The angle brackets denote the Euclidean inner product. This form emphasizes that the filter response is a single dot product per orientation per pixel.

== Matrix Form

Evaluating all $N_s$ orientations simultaneously can be written as a single matrix--vector product.

$ bold(R) = A bold(f) $

The matrix $A in bb(R)^(N_s times N)$ has one row per candidate orientation, where $N$ is the total number of unique pixel positions across all stencils. The vector $bold(f) in bb(R)^N$ contains the image intensities at those positions. Each row of $A$ is a zero-padded version of $bold(alpha)_k^top$, with nonzero entries only at the columns corresponding to that orientation's stencil positions. The winning orientation is then $k^* = arg max_k |R_k|$ applied to the entries of $bold(R)$.

== Convolution Form

Embedding the fused weights into a 2D kernel converts the filter into a standard spatial convolution.

$ R_k (X_0, Y_0) = (h_k star.op f)(X_0, Y_0) $

The kernel $h_k$ is a sparse two-dimensional array with the value $alpha_(k,ell)$ placed at offset $(tilde(delta)_ell^x, tilde(delta)_ell^y)$ and zeros elsewhere. The operator $star.op$ denotes cross-correlation. This form maps directly to GPU primitives such as #text(font: "DejaVu Sans Mono", size: 9pt)[conv2d] and makes clear that the fused stencil is a standard convolutional filter, compatible with existing deep learning and image processing frameworks without modification.

== Tensor Contraction Form

Processing the full image across all orientations simultaneously can be written as a batched convolution producing a three-dimensional response volume.

$ G = cal(H) star.op f $

The filter bank $cal(H) in bb(R)^(N_s times H times W)$ stacks all $N_s$ oriented kernels $h_k$ along a leading dimension, where $H times W$ is the spatial extent of the largest stencil. The output $G in bb(R)^(N_s times N_h times N_v)$ is a response volume indexed by orientation and spatial position. The per-pixel gradient magnitude and edge angle are extracted pointwise as follows.

$ k^* (X_0, Y_0) = arg max_k |G_(k, X_0, Y_0)| $

$ "magnitude" = |G_(k^*, X_0, Y_0)|, quad "angle" = theta_(k^*) $

This form is the natural representation for a fully parallelized GPU implementation, where the entire filter bank is applied as a single batched convolution and the orientation selection reduces to a pointwise argmax over the leading axis.
