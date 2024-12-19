import logging
from pathlib import Path

import numpy as np
import yaml

from derotation.analysis.full_derotation_pipeline import FullPipeline
from derotation.analysis.incremental_derotation_pipeline import (
    IncrementalPipeline,
)


def update_config_paths(
    config, tif_path, bin_path, dataset_path, output_folder, kind="full"
):
    # Set config paths based on provided arguments
    config["paths_read"]["path_to_randperm"] = str(
        Path(dataset_path).parent / "stimlus_randperm.mat"
    )
    config["paths_read"]["path_to_aux"] = str(bin_path)
    config["paths_read"]["path_to_tif"] = str(tif_path)

    # Set output paths to the specified output_folder
    config["paths_write"]["debug_plots_folder"] = str(
        Path(output_folder) / "derotation" / f"debug_plots_{kind}"
    )
    config["paths_write"]["logs_folder"] = str(
        Path(output_folder) / "derotation" / "logs"
    )
    config["paths_write"]["derotated_tiff_folder"] = str(
        Path(output_folder) / "derotation"
    )
    config["paths_write"]["saving_name"] = f"derotated_{kind}.tif"

    return config


def derotate(dataset_folder: Path, output_folder):
    this_module_path = Path(__file__).parent

    # INCREMENTAL DEROTATION PIPELINE
    # find tif and bin files
    bin_path = list(dataset_folder.rglob("*rotation*increment*001.bin"))[0]
    tif_path = list(dataset_folder.rglob("rotation_increment_00001.tif"))[0]

    # Load the config template and update paths

    config_template_path = this_module_path / Path(
        "config/incremental_rotation.yml"
    )
    with open(config_template_path, "r") as f:
        config_incremental = yaml.safe_load(f)

    config_incremental = update_config_paths(
        config_incremental,
        tif_path,
        bin_path,
        dataset_folder,
        output_folder,
        kind="incremental",
    )

    # Create output directories if they don't exist
    Path(config_incremental["paths_write"]["debug_plots_folder"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config_incremental["paths_write"]["logs_folder"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config_incremental["paths_write"]["derotated_tiff_folder"]).mkdir(
        parents=True, exist_ok=True
    )

    # FULL DEROTATION PIPELINE
    # find tif and bin files
    bin_path = list(dataset_folder.rglob("*rotation_*001.bin"))[0]
    tif_path = list(dataset_folder.rglob("rotation_00001.tif"))[0]

    # Load the config template and update paths

    config_template_path = this_module_path / Path("config/full_rotation.yml")
    with open(config_template_path, "r") as f:
        config = yaml.safe_load(f)

    config = update_config_paths(
        config, tif_path, bin_path, dataset_folder, output_folder, kind="full"
    )

    # Create output directories if they don't exist
    Path(config["paths_write"]["debug_plots_folder"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["paths_write"]["logs_folder"]).mkdir(
        parents=True, exist_ok=True
    )
    Path(config["paths_write"]["derotated_tiff_folder"]).mkdir(
        parents=True, exist_ok=True
    )

    logging.info("Running full derotation pipeline")

    # Run the pipeline
    try:
        incremental_derotator = IncrementalPipeline(config_incremental)
        incremental_derotator()
        center = incremental_derotator.center_of_rotation
        ellipse_fits = incremental_derotator.all_ellipse_fits

        derotator = FullPipeline(config)
        derotator.center_of_rotation = center
        if ellipse_fits["a"] < ellipse_fits["b"]:
            rotation_plane_angle = np.degrees(
                np.arccos(ellipse_fits["a"] / ellipse_fits["b"])
            )
            rotation_plane_orientation = np.degrees(ellipse_fits["theta"])
        else:
            rotation_plane_angle = np.degrees(
                np.arccos(ellipse_fits["b"] / ellipse_fits["a"])
            )
            theta = ellipse_fits["theta"] + np.pi / 2
            rotation_plane_orientation = np.degrees(theta)

        rotation_plane_angle = np.round(rotation_plane_angle, 1)
        rotation_plane_orientation = np.round(rotation_plane_orientation, 1)
        derotator.rotation_plane_angle = rotation_plane_angle
        derotator.rotation_plane_orientation = rotation_plane_orientation

        derotator()

        logging.info("Full derotation pipeline complete")

        mean_images = derotator.calculate_mean_images(
            derotator.masked_image_volume, round_decimals=0
        )
        debug_plots_folder = Path(config["paths_write"]["debug_plots_folder"])
        del derotator
        return mean_images, debug_plots_folder
    except Exception as e:
        logging.error("Full derotation pipeline failed")
        logging.error(e.args)
        raise e
