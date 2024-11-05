import argparse
from pathlib import Path

import yaml

from derotation.analysis.full_derotation_pipeline import FullPipeline


def update_config_paths(
    config, tif_path, bin_path, dataset_path, output_folder
):
    # Set config paths based on provided arguments
    config["paths_read"]["path_to_randperm"] = str(
        Path(dataset_path) / "stimlus_randperm.mat"
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


def main(dataset_path, tif_path, bin_path, output_folder):
    # Load the config template and update paths
    config_template_path = Path("derotation/config/full_rotation.yml")
    with open(config_template_path, "r") as f:
        config = yaml.safe_load(f)

    config = update_config_paths(
        config, tif_path, bin_path, dataset_path, output_folder
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

    # Run the pipeline
    derotate = FullPipeline(config)
    derotate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Full Derotation Pipeline"
    )
    parser.add_argument(
        "dataset_path", type=str, help="Path to the dataset's root directory"
    )
    parser.add_argument(
        "tif_path", type=str, help="Full path to the correct TIFF file"
    )
    parser.add_argument(
        "bin_path", type=str, help="Full path to the correct BIN file"
    )
    parser.add_argument(
        "output_folder", type=str, help="Directory for storing output files"
    )
    args = parser.parse_args()

    main(args.dataset_path, args.tif_path, args.bin_path, args.output_folder)
