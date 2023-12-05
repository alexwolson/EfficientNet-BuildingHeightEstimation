#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

# Standard library imports
import argparse
from pathlib import Path

# Third-party library imports
import pandas as pd
from rich.console import Console
from rich.progress import Progress
import google_streetview.api as gsv

# Console object used for pretty-printing progress of capture
console = Console()


def load_api_key(filename: str) -> str | None:
    """Loads the API key from the given file."""
    try:
        with open(filename, "r") as file:
            api_key = file.read().strip()
        return api_key
    except FileNotFoundError:
        console.log(f"File {filename} not found!", style="bold red")
        return None


# Load Street View API key
street_key_path = "keys/street.key"
street_api_key = load_api_key(street_key_path)


def get_streetview(lat: str, lon: str, recordid: int) -> None:
    loc = f"{lat},{lon}"
    param = {
        "size": "640x640",  # max 640x640 pixels
        "location": loc,
        "fov": 90,
        "key": street_api_key,
        "source": "outdoor",
    }

    # console.log(
    #     f"Submitting and downloading streetview images for {recordid} (narrow)..."
    # )
    results = gsv.results([param])
    results.download_links(f"../data/external/massings/{recordid}")


def load_completed_buildings():
    completed_building_path = Path(f"../data/interim/completed_buildings_massings.txt")
    if not completed_building_path.exists():
        completed_building_path.touch()
        Path(f"../data/external/massings").mkdir(parents=True, exist_ok=True)

    with open(f"../data/interim/completed_buildings_massings.txt", "r+") as f:
        return set([line.strip() for line in f.readlines()])


def save_completed_building(building_id):
    with open(f"../data/interim/completed_buildings_massings.txt", "a") as f:
        f.write(f"{building_id}\n")


def main(data_file: str, number: int) -> None:
    buildings_data = pd.read_pickle(data_file)
    completed_buildings = load_completed_buildings()

    console.log(f"{len(completed_buildings)} buildings already photographed")
    console.log(f"{len(buildings_data)} buildings available")

    number -= len(completed_buildings)
    if number > len(buildings_data):
        number = len(buildings_data)

    if number <= 0:
        console.log(f"All requested buildings have been photographed!")
        return

    console.log(f"Capturing {number} more buildings...")

    remaining_buildings = buildings_data[~buildings_data.index.isin(completed_buildings)].sample(frac=1)
    total_buildings = len(remaining_buildings)
    completed_count = 0

    with Progress() as progress:
        task = progress.add_task(f"[cyan]Processing...", total=number)

        for building_id, building in remaining_buildings.iterrows():
            try:
                get_streetview(building["LATITUDE"], building["LONGITUDE"], building_id)
                completed_buildings.add(building_id)
                save_completed_building(building_id)
                completed_count += 1
                progress.update(task, advance=1)

            except Exception as e:
                console.log(f"Error: {str(e)}")
                total_buildings -= 1
                progress.update(task, advance=0)

            if completed_count >= number:
                break

    console.log(f"Completed: {completed_count}/{total_buildings}")

    buildings_data = pd.read_pickle(data_file)

    labels = []
    indices = []
    for building_id in Path("../data/external").glob(f"massings/*"):
        record = buildings_data.loc[int(building_id.name)]
        labels.append(record.to_dict())
        indices.append(int(building_id.name))
    labels = pd.DataFrame(labels, index=indices)
    labels.to_csv(f"../data/external/massings/labels.csv", index=True)

    console.log(f"Saved {len(labels)} labels to ../data/external/massings/labels.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture StreetView and Satellite Images."
    )
    parser.add_argument(
        "--data_file", required=True, help="Path to the buildings data file."
    )

    parser.add_argument(
        "--count", type=int, default=1000, help="Number of images to capture."
    )

    args = parser.parse_args()
    main(args.data_file, args.count)
