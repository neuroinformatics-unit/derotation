(target-derotation)=
# Derotation

A Python package for reconstructing movies of rotating samples acquired with a line scanning microscope.

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
To address certain neuroscience questions in rodents, it might be necessary to image the brain while the head or the body of the animal rotates. In such a case, and even more when the frame rate is low, the acquired movies are distorted by the rotation. These distortions have a peculiar pattern due to the line scanning nature of the microscope, which can be corrected by the derotation package. 

`derotation` provides a set of tools to undo these distortions:
- Recover calcium imaging movies by **line-by-line derotation** that can be fed into standard analysis pipelines such as suite2p;
- Estimate the **center of rotation** using ellipse fitting or Bayesian optimization;
- Validate improvements to the derotation algorithm and pipelines using synthetic data;
- Use debugging plots and logs to verify the quality of the derotation;
- Batch-process multiple datasets with consistent configuration files.


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

