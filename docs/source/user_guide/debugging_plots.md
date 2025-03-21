(user_guide/debugging_plots)=
# Debugging Plots

Derotation provides several plots to help you inspect each stage of the pipeline. These are essential for verifying that:
- Analog signals were correctly parsed
- Rotation angles are sensible
- The center of rotation is well estimated
- The derotation result is geometrically coherent

All plots are saved automatically to the `debug_plots_folder` specified in your config.

---

## 📈 Signal diagnostics
These plots visualize the raw analog signals and detected events:
- Frame and line clock pulses
- Rotation on/off signal
- Rotation ticks

Useful for verifying:
- That all signals are present
- That tick peaks are cleanly detected
- That frames and rotations don’t overlap unexpectedly

---

## 🌀 Angle interpolation
These plots show the rotation angle over time, interpolated per frame or line:
- Continuous progression across frames
- Discontinuities or gaps in the ticks

They help spot problems like missing ticks or malformed rotations.

---

## 🎯 Center estimation
Depending on the method used:
- **Ellipse fitting**: shows blob detections, tracked positions, and fitted ellipse
- **Bayesian optimization**: shows the PTD landscape, optimization steps, and chosen center

Helpful for evaluating whether the chosen center is valid and consistent.

---

## 🖼️ Image stack visualizations
These include:
- Sample input frames
- Derotated frames side-by-side
- PTD maps before and after derotation
- Max intensity projections of the output stack

You can quickly scan these to catch geometric inconsistencies or residual motion.

---

## 🔧 Custom plotting hooks
Advanced users can inject custom plotting logic using hooks:
```python
from derotation.plotting_hooks.for_derotation import image_completed, line_addition

hooks = {
    "plotting_hook_line_addition": line_addition,
    "plotting_hook_image_completed": image_completed,
}

pipeline = FullPipeline(config)
pipeline.hooks = hooks
pipeline()
```

> ⚠️ Note: Hooks may slow down processing significantly. Use them for inspection only.

---

## Best practices
- Always check signal parsing plots before trusting the derotation
- Use center diagnostics to understand errors in reconstruction
- Compare before/after PTD maps to evaluate alignment quality
- Keep `debugging_plots: True` in your config until the pipeline is fully tuned

Plots are saved with descriptive filenames and ordered by step. You can safely delete them when confident in your setup.

