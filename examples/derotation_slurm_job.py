import os
import sys
from pathlib import Path

import yaml
from full_derotation_pipeline import FullPipeline
from incremental_derotation_pipeline import (
    IncrementalPipeline,
)

# =====================================================================
# Set up the config files
# =====================================================================

job_id = int(sys.argv[1:][0])
dataset_path = sys.argv[1:][1]
datasets = [path for path in os.listdir(dataset_path) if path.startswith("23")]
dataset = datasets[job_id]

bin_files = [
    file
    for file in os.listdir(f"{dataset_path}/{dataset}/aux_stim/")
    if file.endswith(".bin")
]
full_rotation_bin = [file for file in bin_files if "_rotation" in file][0]
incremental_bin = [file for file in bin_files if "increment" in file][0]

image_files = [
    file
    for file in os.listdir(f"{dataset_path}/{dataset}/imaging/")
    if file.endswith(".tif")
]
full_rotation_image = [file for file in image_files if "rotation_0" in file][0]
incremental_image = [file for file in image_files if "increment_0" in file][0]

Path(f"{dataset_path}/{dataset}/debug_plots_incremental/").mkdir(
    parents=True, exist_ok=True
)
Path(f"{dataset_path}/{dataset}/debug_plots_full/").mkdir(
    parents=True, exist_ok=True
)
Path(f"{dataset_path}/{dataset}/logs/").mkdir(parents=True, exist_ok=True)
Path(f"{dataset_path}/{dataset}/derotated/").mkdir(parents=True, exist_ok=True)

for config_name in ["incremental_rotation", "full_rotation"]:
    with open(f"derotation/config/{config_name}.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["paths_read"][
        "path_to_randperm"
    ] = f"{dataset_path}/stimlus_randperm.mat"
    bin_name = (
        incremental_bin
        if config_name == "incremental_rotation"
        else full_rotation_bin
    )
    config["paths_read"][
        "path_to_aux"
    ] = f"{dataset_path}/{dataset}/aux_stim/{bin_name}"
    image_name = (
        incremental_image
        if config_name == "incremental_rotation"
        else full_rotation_image
    )
    config["paths_read"][
        "path_to_tif"
    ] = f"{dataset_path}/{dataset}/imaging/{image_name}"
    config["paths_write"][
        "debug_plots_folder"
    ] = f"{dataset_path}/{dataset}/debug_plots_{config_name.split('_')[0]}"
    config["paths_write"]["logs_folder"] = f"{dataset_path}/{dataset}/logs/"
    config["paths_write"][
        "derotated_tiff_folder"
    ] = f"{dataset_path}/{dataset}/derotated/"
    config["paths_write"][
        "saving_name"
    ] = f"derotated_image_stack_{config_name.split('_')[0]}"

    with open(f"derotation/config/{config_name}_{job_id}.yml", "w") as f:
        yaml.dump(config, f)


# =====================================================================
# Run the pipeline
# =====================================================================

derotate_incremental = IncrementalPipeline(f"incremental_rotation_{job_id}")
derotate_incremental()

derotate_full = FullPipeline(f"full_rotation_{job_id}")
derotate_full.mask_diameter = derotate_incremental.new_diameter
derotate_full()
