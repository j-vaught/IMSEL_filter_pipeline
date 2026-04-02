= Experimental Setup

This section describes the datasets, evaluation metrics, comparison methods, and experimental protocols used to assess the Anisotropic Stencil Filter and its geometric kernel variants against both classical gradient operators and state-of-the-art deep learning models.

== Datasets

Three edge detection benchmarks spanning distinct imaging domains compose the evaluation suite, totaling 280 test images.

The Berkeley Segmentation Dataset (BSDS500) @arbelaez2011bsds500 provides 200 test images at $481 times 321$ resolution, each annotated by multiple human observers who marked object-level boundaries. BSDS500 remains the standard benchmark for contour detection research. Its ground truth encodes _semantic_ boundaries, meaning that annotators traced object contours rather than marking every photometric intensity transition. This distinction is important because it inherently disadvantages non-learned methods, which respond to all intensity gradients regardless of semantic significance. A classical filter that faithfully detects texture edges inside an object will be penalized for producing false positives against a ground truth that omits those edges.

The Barcelona Images for Perceptual Edge Detection dataset (BIPED v1) @poma2020biped consists of 50 test images at $1280 times 720$ resolution depicting outdoor urban and suburban scenes. The annotations emphasize fine structural detail, including thin wires, sign edges, and architectural features, at a resolution approximately five times greater than BSDS500. BIPED tests whether a method can maintain edge localization accuracy on larger images where small-scale structures occupy only a few pixels and where the total number of edge pixels per image is substantially higher.

The Underwater Edge Detection (UDED) dataset contains 30 images at $339 times 510$ resolution captured in underwater environments. These images exhibit the characteristic degradation of subaqueous imaging @schettini2010underwater, including color cast from wavelength-dependent attenuation, reduced contrast from suspended particulate scattering, and non-uniform illumination from artificial light sources. UDED tests generalization to imaging conditions that differ markedly from the natural photographic scenes on which deep learning models are typically trained.

== Evaluation Protocol

All methods are evaluated using the standard BSDS500 evaluation protocol. For each method, the gradient magnitude map is thresholded at 1001 uniformly spaced levels spanning the full output range. At each threshold, a binary edge map is produced and compared against the ground truth using precision and recall with a matching tolerance of 3 pixels. This tolerance accommodates slight localization differences between predicted and annotated edge positions, which is standard practice given the inherent subjectivity in ground truth placement.

Two summary metrics are reported. The Optimal Dataset Scale (ODS) F-score selects the single threshold that maximizes the F-score across all images in the dataset simultaneously, representing the best fixed-threshold performance. The Optimal Image Scale (OIS) F-score selects the best threshold independently for each image and averages the resulting per-image F-scores, representing the upper bound achievable with a perfect per-image threshold oracle. The F-score is the harmonic mean of precision $P$ and recall $R$,

$ F = (2 P R) / (P + R) $ <eq:fscore>

which penalizes methods that sacrifice one metric for the other.

== Comparison Methods

We compare the proposed filter against 54 classical filter configurations and 5 deep learning models.

=== Classical Filters

The classical comparison set includes the Sobel operator at kernel sizes $3 times 3$ through $11 times 11$, the Prewitt operator at $3 times 3$ through $5 times 5$, the Scharr $3 times 3$ optimized kernel @scharr2000optimal, the Roberts Cross $2 times 2$ diagonal operator, and the Kirsch compass operator evaluated over 8 directional kernels at $45 degree$ spacing. Scale-space methods include Gaussian first derivatives at scales $sigma in {1.0, 1.5, 2.0}$, the Laplacian of Gaussian @marr1980log, and the Difference of Gaussians approximation. Steerable filters @freeman1991steerable are evaluated at both first and second order. The Frei-Chen basis set @freichen1977fast, which decomposes local structure into edge, line, and isotropic subspaces, is also included. Each classical method is evaluated at its best parameter configuration per dataset, ensuring a fair comparison that reflects the strongest achievable performance rather than a single default setting.

=== Deep Learning Models

Five deep learning architectures represent the current state of the art in learned edge detection. DexiNed @poma2020dexined uses a dense extreme inception network that was trained from scratch on edge detection data without ImageNet pre-training. TEED @soria2023teed is a tiny and efficient architecture designed for real-time inference with minimal parameter count. PIDINet @su2021pidinet incorporates pixel difference convolutions that embed classical gradient priors directly into the network layers. NBED @chen2024nbed is a neural-network-based boundary extractor with explicit thin-edge preservation mechanisms. DiffusionEdge @ye2024diffusionedge formulates edge detection as a conditional denoising diffusion process, generating edge maps through iterative refinement. All deep learning models are evaluated using their published pretrained weights without any fine-tuning on the test datasets, ensuring that the comparison reflects off-the-shelf generalization rather than domain-adapted performance.

== Proposed Filter Configurations

Three variants of the anisotropic filter are evaluated, each representing a different strategy for constructing the orientation-dependent weight envelope.

The fused polynomial stencil (ASF) is evaluated over a parameter grid spanning neighborhood sizes $N_p in {25, 50, 100, 250}$, polynomial degrees $d in {2, 4}$, line extension half-widths $m in {0, 1, 2, 7, 14}$, and orientation counts $N_s in {4, 6, 12, 18, 36}$. The best-performing configuration is reported separately for each dataset, as optimal parameters depend on image resolution and noise characteristics.

The rectangular Gaussian kernel variant defines the anisotropic envelope as a rectangular footprint aligned with the candidate edge direction. The parameter sweep covers tangential scales $sigma_u in {1.5, 2.0, 2.5, 3.0}$, normal scales $sigma_v in {0.8, 1.0, 1.2, 1.5}$, and orientation counts $N_s in {18, 36}$, with a fixed $15 times 15$ grid size.

The elliptical Gaussian kernel variant replaces the rectangular footprint with an elliptical Gaussian envelope whose major axis is aligned tangentially to the candidate edge direction. The parameter sweep is identical to the rectangular variant ($sigma_u$, $sigma_v$, $N_s$, and grid size), enabling direct comparison of the two geometric alternatives under matched conditions.

== Noise Robustness Protocol

Robustness to noise is evaluated systematically across five corruption types that represent common sources of image degradation. Additive white Gaussian noise models electronic sensor noise. Salt-and-pepper noise simulates dead or saturated pixels from sensor defects. Poisson noise models the photon-counting statistics of low-light imaging. Speckle noise (multiplicative) represents the coherent interference patterns found in radar and ultrasound imagery. Uniform noise serves as a baseline additive corruption with bounded amplitude.

For each noise type, seven signal-to-noise ratio (SNR) levels are tested, ranging from 0.1 (severe corruption where the noise power exceeds the signal by a factor of 10) to 5.0 (mild corruption), plus a clean baseline. The noise is applied to a representative subset of 5 images per dataset (20 images total, sampled to span diverse scene content), and all methods are evaluated at each noise condition. The ODS F-score is reported as a function of SNR for each method, producing degradation curves that reveal both the absolute noise tolerance and the rate of performance decline for each approach.

== Hardware and Software Environment

All GPU experiments are conducted on an NVIDIA A100-SXM4-40GB accelerator. CPU preprocessing and classical filter evaluation use an AMD EPYC 7763 processor with 64 cores. Deep learning inference is performed on the same A100 GPU using PyTorch 2.1 with CUDA 12.1. The proposed ASF variants use a custom Triton kernel compiled against the same CUDA toolkit. All timing measurements are averaged over 10 runs with a 3-run warm-up to eliminate JIT compilation overhead and ensure stable GPU clock frequencies.
