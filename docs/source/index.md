(target-derotation)=
# Derotation

A Python package for reconstructing images of rotating samples acquired with a line scanning microscope.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 📘 User guide
:link: user_guide/index
:link-type: doc

Installation, configuration, supported formats, and key concepts.
:::

:::{grid-item-card} 🧪 Examples
:link: examples/index
:link-type: doc

A gallery of real and synthetic examples using `derotation`.
:::

:::{grid-item-card} 💬 Join the development
:link: community/index
:link-type: doc

How to get in touch, ask questions, and contribute.
:::
::::

![](_static/derotation_overview.png)

## Overview

Many neuroscience experiments involve rotating samples under a microscope. These rotations, while essential, can introduce spatial distortions into recorded image stacks.

`derotation` provides a set of tools to undo these distortions:
- Recover interpretable images by **line-by-line derotation**.
- Estimate the **center of rotation** using ellipse fitting or Bayesian optimization.
- Validate your process using synthetic data and debugging plots.
- Batch-process multiple datasets with consistent configuration files.

Check out our [mission and scope](target-mission) and our [roadmap](target-roadmaps) for more on the project's direction.

```{include} ../../README.md
:start-after: '## Citation'
:end-before: '## License'
```

```{toctree}
:maxdepth: 2
:hidden:

user_guide/index
examples/index
community/index
api_index
```

