(target-derotation)=
# Derotation
A Python package for reconstructing movies of rotating samples acquired with a line scanning microscope.

```{figure} _static/mean_images_with_incremental.png

On the left, the mean image of a 3-photon movie in which the head of the animal was rotating. In the center, the mean image after derotation, and on the left the mean image of the derotated movie after suite2p registration. As you can see, already after derotation the cells are visible and have well defined shapes.
```

::::{grid} 2
:gutter: 4

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

::::

## Overview
```{include} ../../README.md
:start-after: '## Overview'
:end-before: '## Data Source & Funding'
```

## Data Source & Funding
```{include} ../../README.md
:start-after: '## Data Source & Funding'
```

```{toctree}
:maxdepth: 2
:hidden:

user_guide/index
examples/index
contributing
api_index
```
