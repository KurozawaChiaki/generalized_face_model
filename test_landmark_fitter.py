import argparse
import json
from src.landmark_fitter import LandmarkFitter


def main():
    # Load configs
    general_config: dict = json.load(open("config/general.json"))
    fitting_config = json.load(open("config/sculptor_fitting.json"))

    target_mesh_path: str = fitting_config["target_mesh_path"]
    sculptor_path: str = general_config["sculptor_config"]["model_path"]

    # Initialize landmark fitter
    landmark_fitter = LandmarkFitter(
        target_path=target_mesh_path,
        sculptor_paradict_path=sculptor_path,
        general_config=general_config,
        sculptor_scale=1000,
        fitting_method="mesh",
        output_dir=fitting_config["output_path"],
        base_iterations=1000,
        lr=0.001,
    )

    landmark_fitter.fit()


if __name__ == "__main__":
    main()
