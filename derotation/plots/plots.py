from find_centroid import (
    in_region,
    not_center_of_image,
)
from matplotlib import pyplot as plt


def plot_drift_of_centroids(
    centers, centers_rotated, centers_rotated_corrected
):
    #  plot drift of centers
    fig, ax = plt.subplots(3, 1)
    for k, centroid in enumerate(centers):
        for c in centroid:
            if not_center_of_image(c) and in_region(c):
                ax[0].plot(k, c[1], marker="o", color="red")
                ax[0].plot(k, c[0], marker="o", color="blue")
                # ax[0].set_ylim(80, 180)
    for k, centroid in enumerate(centers_rotated):
        for c in centroid:
            if not_center_of_image(c) and in_region(c):
                ax[1].plot(k, c[1], marker="o", color="red")
                ax[1].plot(k, c[0], marker="o", color="blue")
                # ax[0].set_ylim(80, 180)
    for k, centroid in enumerate(centers_rotated_corrected):
        for c in centroid:
            if not_center_of_image(c) and in_region(c):
                ax[2].plot(k, c[1], marker="o", color="red")
                ax[2].plot(k, c[0], marker="o", color="blue")
                # ax[0].set_ylim(80, 180)

    return fig


def derotation_video_with_rotation_plot(
    rotated_image,
    image,
    rotated_image_corrected,
    centers,
    centers_rotated,
    centers_rotated_corrected,
    frames_start,
    signed_rotation_degrees,
    image_rotation_degree_per_frame,
):
    # Create a figure and axis for displaying the images
    fig, ax = plt.subplots(1, 4)

    ax[2].set_title("Rotation degrees per frame")

    # Iterate through each image
    for i, (image_rotated, image_original, image_corrected) in enumerate(
        zip(rotated_image, image, rotated_image_corrected)
    ):
        ax[0].imshow(image_original, cmap="gist_ncar")
        ax[1].imshow(image_rotated, cmap="gist_ncar")
        ax[2].imshow(image_corrected, cmap="gist_ncar")

        for c in centers[i]:
            if not_center_of_image(c) and in_region(c):
                # dim blob
                ax[0].plot(c[1], c[0], marker="*", color="red")
            if not not_center_of_image(c):
                # bright blob
                ax[0].plot(c[1], c[0], marker="*", color="white")
        for c in centers_rotated[i]:
            if not_center_of_image(c) and in_region(c):
                # dim blob
                ax[1].plot(c[1], c[0], marker="*", color="red")
            if not not_center_of_image(c):
                # bright blob
                ax[1].plot(c[1], c[0], marker="*", color="white")
        for c in centers_rotated_corrected[i]:
            if not_center_of_image(c) and in_region(c):
                # dim blob
                ax[2].plot(c[1], c[0], marker="*", color="red")
            if not not_center_of_image(c):
                # bright blob
                ax[2].plot(c[1], c[0], marker="*", color="white")

        ax[0].axis("off")
        ax[1].axis("off")
        ax[2].axis("off")

        #  add a vertical line on the plot on ax 2
        ax[3].axvline(frames_start[i], color="black", linestyle="--")
        ax[3].plot(signed_rotation_degrees, label="rotation degrees")
        ax[3].plot(
            frames_start,
            image_rotation_degree_per_frame,
            linestyle="none",
            marker="o",
            color="red",
        )

        plt.pause(0.001)
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()
        ax[3].clear()

    ax[0].set_title("Original image")
    ax[1].set_title("Rotated image")
    ax[2].set_title("Corrected image")
    ax[3].set_title("Rotation degrees per frame")

    # axis off for the first two plots
    ax[0].axis("off")
    ax[1].axis("off")

    return fig


def threshold_boxplot(diffs, threshold):
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.boxplot(diffs)
    ax.set_title("Threshold to identify frames start and end")
    ax.set_ylabel("Difference between frames")

    ax.axhline(threshold, 0, len(diffs), color="red", label="threshold")
    ax.axhline(-threshold, 0, len(diffs), color="red", label="threshold")

    return fig


def analog_signals_overview_plots(
    diffs,
    frame_clock,
    frames_start,
    frames_end,
    line_clock,
    full_rotation,
    rotation_on,
    rotation_ticks,
    rotation_ticks_peaks,
):
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.plot(diffs, label="frame clock", color="black", alpha=0.5)

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(
        frame_clock,
        label="frame clock",
        color="black",
        alpha=0.5,
        rasterized=True,
    )
    # plot dots for starting and ending points of the frame_clock signal
    ax[0].plot(
        frames_start,
        frame_clock[frames_start],
        linestyle="none",
        marker="o",
        color="red",
        alpha=0.5,
        rasterized=True,
    )
    ax[0].plot(
        frames_end,
        frame_clock[frames_end],
        linestyle="none",
        marker="o",
        color="green",
        alpha=0.5,
        rasterized=True,
    )
    ax[1].plot(
        line_clock,
        label="line clock",
        color="red",
        alpha=0.5,
        rasterized=True,
    )
    ax[2].plot(
        full_rotation,
        label="rot tick",
        color="blue",
        alpha=0.5,
        rasterized=True,
    )
    ax[2].plot(
        rotation_on,
        label="rotation with direction, 1 = CW, -1 = CCW",
        color="green",
        alpha=0.5,
        rasterized=True,
    )
    ax[3].plot(
        rotation_ticks,
        label="rot tick 2",
        color="green",
        alpha=0.5,
        marker="o",
        rasterized=True,
    )
    ax[3].plot(
        rotation_ticks_peaks,
        # np.ones(len(rot_tick2_peaks)) * 5.2,
        rotation_ticks[rotation_ticks_peaks],
        linestyle="none",
        marker="*",
        color="red",
        alpha=0.5,
        rasterized=True,
    )

    # set the initial x axis limits
    for axis in ax:
        # axis.set_xlim(1610000, 1800000)
        # axis.set_xlim(1680000, 1710000)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    #  set subplots titles
    ax[0].set_title(
        "Frame clock (black) and starting/ending points (red/green)"
    )
    ax[1].set_title("Line clock")
    ax[2].set_title("Full rotation info")
    ax[3].set_title("Rotation ticks, 0.2 deg for 1 tick (green), peaks (red)")

    #  plot title
    fig.suptitle("Frame clock and rotation ticks")

    return fig
