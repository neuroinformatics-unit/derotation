import logging
from pathlib import Path

import yaml

from derotation.analysis.full_derotation_pipeline import FullPipeline


def update_config_paths(
    config, tif_path, bin_path, dataset_path, output_folder
):
    # Set config paths based on provided arguments
    config["paths_read"]["path_to_randperm"] = str(
        Path(dataset_path).parent / "stimlus_randperm.mat"
    )
    config["paths_read"]["path_to_aux"] = str(bin_path)
    config["paths_read"]["path_to_tif"] = str(tif_path)

    # Set output paths to the specified output_folder
    config["paths_write"]["debug_plots_folder"] = str(
        Path(output_folder) / "debug_plots_full"
    )
    config["paths_write"]["logs_folder"] = str(Path(output_folder) / "logs")
    config["paths_write"]["derotated_tiff_folder"] = str(
        Path(output_folder) / "derotated"
    )
    config["paths_write"]["saving_name"] = "derotated_image_stack_full"

    return config


def derotate(dataset_folder: Path, output_folder):
    # find tif and bin files
    bin_path = list(dataset_folder.rglob("*rotation_*001.bin"))[0]
    tif_path = list(dataset_folder.rglob("rotation_00001.tif"))[0]

    # Load the config template and update paths
    this_module_path = Path(__file__).parent
    config_template_path = this_module_path / Path("config/full_rotation.yml")
    with open(config_template_path, "r") as f:
        config = yaml.safe_load(f)

    config = update_config_paths(
        config, tif_path, bin_path, dataset_folder, output_folder
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
        derotator = FullPipeline(config)
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
