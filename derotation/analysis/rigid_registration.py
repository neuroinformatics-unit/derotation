#  Register the images to a common reference frame
#  using Fourier-Mellin transform


import imreg_dft


def refine_derotation(images):
    reference_image = images[0]
    output = []
    for i in range(0, len(images)):
        output.append(
            imreg_dft.imreg.similarity(
                reference_image,
                images[i],
                constraints={
                    "angle": [0, 3],
                    "scale": [1, 0],
                    "tx": [0, 0],
                    "ty": [0, 0],
                },
            )
        )
        print(f"angle: {output[i]['angle']}, scale: {output[i]['scale']}")
    return output
