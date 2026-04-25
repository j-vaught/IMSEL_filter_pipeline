# NMS/GMM Edge Detector

This subpackage implements the proposed multi-scale, multi-domain edge detector
from the NMS/GMM reference paper. It builds LF response stacks across selected
image domains and filter half-widths, refines each scale/domain orientation with
periodic cubic splines, fuses the stack with a weighted two-component Gaussian
mixture model, and applies denser non-maximum suppression with Canny-style
hysteresis.

```python
import imageio.v3 as iio

from edgecritic.nms_gmm import NMSGMMConfig, detect_edges

image = iio.imread("input.png")
config = NMSGMMConfig(
    half_widths=(3, 7, 11),
    domains="auto",
    n_orientations=36,
)

result = detect_edges(image, config=config)
edges = result.edges
```

The command line entry point can be run as a module. The CLI writes a binary
edge map and can optionally write the thinned NMS magnitude map.

```bash
PYTHONPATH=src python -m edgecritic.nms_gmm.cli input.png edges.png --save-nms nms.png
```

The core package depends only on NumPy and SciPy. The CLI loads and saves images
with `imageio`, which is imported lazily so the library API stays lightweight.

For low-contrast aquatic scenes with strong water texture, use the stricter
continuity-oriented preset.

```python
config = NMSGMMConfig.aquatic()
result = detect_edges(image, config=config)
```
