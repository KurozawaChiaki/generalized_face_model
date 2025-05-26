import argparse
import json
from src.sculptor_fitter import SculptorFitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True)

    args = parser.parse_args()
    general_config: dict = json.load(open("config/general.json"))
    
    if args.f == "sculptor_fitting":
        fitting_config = json.load(open("config/sculptor_fitting.json"))
        target_mesh_path: str = fitting_config["target_mesh_path"]
        sculptor_path: str = general_config["sculptor_config"]["model_path"]

        sculptor_fitter = SculptorFitter(target_mesh_path=target_mesh_path, 
                                        sculptor_paradict_path=sculptor_path, 
                                        num_iterations=1,
                                        output_dir=fitting_config["output_path"])
        sculptor_fitter.fit()
    else:
        raise ValueError(f"Invalid fitting method: {args.f}")


if __name__ == "__main__":
    main()