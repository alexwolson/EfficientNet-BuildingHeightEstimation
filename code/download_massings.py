import zipfile
from pathlib import Path
from TorontoOpenData import TorontoOpenData
import argparse
import geopandas as gpd


def download_massings():
    # make path if it does not exist
    args.output.mkdir(parents=True, exist_ok=True)
    TorontoOpenData().download_dataset("3d-massing", args.output)
    output_directory = args.output / "3d-massing"
    # Unzip the data
    for file in output_directory.iterdir():
        file_name = file.stem
        print(f"Unzipping {file_name}")
        try:
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(args.output / file_name)
        except:
            continue


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default="../data/external")
    args = parser.parse_args()

    data_path = args.output / "3D Massing (WGS84)" / "3DMassingShapefile_2022_WGS84.shp"

    if not data_path.exists():
        download_massings()

    gdf = gpd.read_file(data_path)
    gdf = gdf[gdf.BLDG_SRC == "Photogrammetrics"]
    gdf.drop(columns=["geometry"], inplace=True)
    gdf.to_pickle("../data/processed/3d-massing.pkl")
