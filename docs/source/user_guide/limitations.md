(user_guide/limitations)=
# Limitations

While **Derotation** is a powerful tool for reconstructing images from rotating scans, it is important to be aware of its current limitations. This page outlines the main assumptions, technical constraints, and known gaps.

---

## Current limitations

### 1. **Input format assumptions**
- The image stack must be provided as a **TIFF** file.
- Analog signals are expected in a `.bin` file format with channels ordered as configured.
- A `.mat` file may be required for stimulus metadata (e.g. random permutations).

### 2. **Rigid channel expectations**
The pipeline assumes analog signals follow a standard channel order and naming:
- `camera`
- `scanimage_frameclock`
- `scanimage_lineclock`
- `photodiode2`
- `PI_rotON`
- `PI_rotticks`

Changes to this structure require editing config and possibly modifying signal parsing logic.

### 3. **Pipeline modes are fixed**
The full and incremental pipelines are designed for two distinct experimental setups:
- **Full rotations** (random 360° trials)
- **Incremental steps** (e.g., 0.2°)

Other protocols are not supported without code modifications.

### 4. **Rotation ticks must be step-like**
The pipeline assumes that rotation is driven by a **stepper motor**, producing clean, discrete ticks. Continuous or noisy rotation signals may lead to angle misestimation.

### 5. **Limited motion models**
Derotation only supports **planar rotations around a fixed center**. It does not account for:
- Sample translation
- Deformation
- Non-circular motion

### 6. **Center estimation depends on bright blobs**
Both ellipse fitting and PTD-based optimization rely on visible features in the image stack that rotate cleanly. If your data lacks such structure, automatic center detection may fail.

### 7. **Performance constraints**
- Full Bayesian Optimization for the rotation center can be slow (minutes per dataset).
- Large TIFF stacks can be memory-intensive. Chunking and caching are not yet implemented.

### 8. **Experimental and evolving interface**
- The package is under active development.
- APIs and config formats may change.
- Some modules are not yet documented or tested for public use.

---

## Recommendations

- Start with synthetic data to validate your configuration.
- Visually inspect debugging plots at each step.
- Use known ground-truth datasets to calibrate your parameters.
- Consider bypassing the full pipeline and using only the core `rotate_an_image_array_line_by_line` function if you have externally processed angle data.

---

We're actively working to improve flexibility, support more input formats, and optimize performance. Feedback and issues are very welcome!

