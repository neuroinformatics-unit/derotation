(user_guide/examples/index)=
# Examples

Explore a variety of usage examples to help you understand and test the derotation toolkit. These range from core function usage to full pipelines with debugging, as well as synthetic data generation.

---

## 📦 Minimal usage: Rotate and derotate an image
Use `rotate_an_image_array_line_by_line` to deform an image line-by-line and reconstruct it:
- Start with a static image (e.g., a square or dog)
- Apply synthetic line-by-line rotation
- Recover the original using the core function

Try:
- `rotate_and_derotate_a_square.py`
- `deformation_by_line_of_a_dog.py`

---

## 🌀 Full and incremental derotation pipelines
These scripts demonstrate how to launch the two pipeline modes:
- `derotate.py` → Full rotation protocol
- `derotate_incremental.py` → Stepwise rotation protocol

They use YAML configs and produce logs, plots, and derotated TIFFs.

We recommend loading a template configuration file and overriding the necessary fields:

```python
import yaml
from pathlib import Path

with open(Path("config/full_rotation.yml")) as f:
    config = yaml.safe_load(f)

# Override fields for your dataset
config["paths_read"]["path_to_tif"] = "/my/data/image_stack.tif"
config["paths_read"]["path_to_aux"] = "/my/data/signals.bin"
config["paths_write"]["debug_plots_folder"] = "/my/debug/"

from derotation.analysis.full_derotation_pipeline import FullPipeline
pipeline = FullPipeline(config)
pipeline()
```

---

## 🎛️ Custom hooks and center re-use
Advanced example showing how to:
- Pass plotting hooks
- Reuse center of rotation across pipelines

Try:
- `derotation_with_shifted_center.py`

---

## 🧪 Synthetic image stacks
Simulate rotating blobs using the `Rotator` class and visualize their deformation:
- `elliptical_rotations.py` creates synthetic frames with variable plane orientation.
- Useful for understanding how misaligned planes affect rotation.

---

## 🐶 Fun with real-world images
Try line-by-line deformation of an image like a dog photo, just for fun and testing.
- `deformation_by_line_of_a_dog.py`

---

## 💡 Tips
- Use these scripts to validate your configuration and understand the behavior of core methods.
- Debugging plots and logs are saved alongside outputs.
- For headless or HPC environments, comment out `plt.show()`.

If you want to contribute an example, check out our [contribution guide](../community/index.md).

