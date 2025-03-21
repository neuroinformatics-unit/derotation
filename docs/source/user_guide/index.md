(user_guide/index)=
# User Guide

Welcome to the user guide for **Derotation**.
This guide walks you through the key concepts, configuration, and limitations of the derotation pipeline.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🧠 Concepts
:link: key_concepts
:link-type: doc

Understand how derotation works, from analog signals to line-based transforms.
:::

:::{grid-item-card} 📁 Configuration
:link: configuration
:link-type: doc

Explore how to structure config files and set paths for derotation.
:::

:::{grid-item-card} 📉 Limitations
:link: limitations
:link-type: doc

Current constraints and assumptions of the pipeline.
:::
::::

---

## Quick Start

Install via pip:
```bash
pip install derotation
```

Or create a conda environment:
```bash
conda create -n derotation-env python=3.12
conda activate derotation-env
```

Then try out one of the example scripts:
```bash
python3 examples/derotate.py  # Full rotation
python3 examples/derotate_incremental.py  # Incremental rotation
```

For more details, head over to the topics above!

