import hashlib
import json
from pathlib import Path
import cv2
import pandas as pd
import shapely.geometry
import zipfile
import argparse

import requests
from rich.console import Console
from rich.progress import Progress

console = Console()


def unzip_files(zip_path, output_folder):
    console.log(f"Unzipping {zip_path} to {output_folder}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)
    console.log("Unzipping completed.")


def count_files(folder_path):
    return sum(1 for _ in folder_path.rglob("*_labelIds.png"))


def process_images(
    image_folder, label_folder, coarse_folder, output_base_folder, csv_base_path
):
    console = Console()

    console.log("Creating output folders for train, val, test, and train_extra sets.")
    for partition in ["train", "val", "test", "train_extra"]:
        Path(output_base_folder, partition).mkdir(parents=True, exist_ok=True)
    console.log("Output folders created.")

    for partition in ["train", "val", "test", "train_extra"]:
        console.log(f"Processing {partition} partition.")
        data = []

        if partition == "train_extra":
            partition_label_folder = coarse_folder / partition
            partition_image_folder = image_folder / partition
            suffix = "_gtCoarse_polygons.json"
            partition_output_folder = Path(
                output_base_folder, "train"
            )  # Move to standard "train"
        else:
            partition_label_folder = label_folder / partition
            partition_image_folder = image_folder / partition
            suffix = "_gtFine_polygons.json"
            partition_output_folder = Path(output_base_folder, partition)

        partition_csv_path = csv_base_path / f"{partition}_labels.csv"

        total_files = sum(1 for _ in partition_label_folder.rglob("*")) / 4

        with Progress() as progress:
            task = progress.add_task(
                f"[green]Processing {partition}...", total=total_files
            )

            for city_label_folder in partition_label_folder.iterdir():
                city_image_folder = partition_image_folder / city_label_folder.name

                for label_file in city_label_folder.rglob("*"):
                    if label_file.name.endswith(suffix):
                        with open(label_file) as f:
                            label = json.load(f)

                        building_area = 0
                        for obj in label["objects"]:
                            if obj["label"] == "building":
                                building_area += shapely.geometry.Polygon(
                                    obj["polygon"]
                                ).area

                        building_area /= 2048 * 1024

                        image_filename = label_file.name.replace(
                            suffix, "_leftImg8bit.png"
                        )
                        image_path = city_image_folder / image_filename

                        if not image_path.exists():
                            console.log(f"Image {image_path} does not exist.")
                            exit(1)

                        data.append(
                            {
                                "filename": image_filename.replace(".png", ""),
                                "building_area": building_area,
                            }
                        )

                        destination = partition_output_folder / image_filename
                        destination = destination.with_suffix(".jpg")

                        img = cv2.imread(str(image_path))
                        cv2.imwrite(
                            str(destination), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                        )

                        progress.update(task, advance=1)

        console.log(f"{partition} partition images processed.")
        df = pd.DataFrame(data)
        df.to_csv(partition_csv_path, index=False)
        console.log(f"{partition} partition labels saved to CSV.")


def download_file(session, url, output_path, file_size):
    console.log(f"Attempting to download {url} to {output_path}")

    # Check if the file already exists
    if Path(output_path).exists():
        resume_byte_pos = Path(output_path).stat().st_size
    else:
        resume_byte_pos = 0

    # Initialize the progress bar
    with Progress() as progress:
        task = progress.add_task("[cyan]Downloading...", total=file_size)

        # Set the range header to resume the download
        headers = {"Range": f"bytes={resume_byte_pos}-"}

        with session.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()

            # Update the progress bar to the resume position
            progress.update(task, completed=resume_byte_pos)

            # Check if the file has been completely downloaded
            if resume_byte_pos >= file_size:
                console.log("File already downloaded.")
                return

            console.log(f"Resuming download from byte {resume_byte_pos}.")

            with open(output_path, "ab") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    console.log("Download completed.")


def login_and_download(username, password, dataset):
    session = requests.Session()
    login_data = {"username": username, "password": password, "submit": "Login"}
    session.post("https://www.cityscapes-dataset.com/login/", data=login_data)

    download_params = {
        "gtFine": {
            "path": "../data/interim/gtFine_trainvaltest.zip",
            "size": 241 * 1024 * 1024,
            "packageID": 1,
        },
        "leftImg8bit": {
            "path": "../data/interim/leftImg8bit_trainvaltest.zip",
            "size": 11 * 1024 * 1024 * 1024,
            "packageID": 3,
        },
        "gtCoarse": {
            "path": "../data/interim/gtCoarse.zip",
            "size": 135 * 1024 * 1024,
            "packageID": 2,
        },
        "train_extra": {
            "path": "../data/interim/leftImg8bit_trainextra.zip",
            "size": 44 * 1024 * 1024 * 1024,
            "packageID": 4,
        },
    }

    if dataset not in download_params:
        console.log("Please provide a valid dataset parameter.")
        exit(1)

    param = download_params[dataset]
    download_file(
        session,
        f"https://www.cityscapes-dataset.com/file-handling/?packageID={param['packageID']}",
        param["path"],
        param["size"],
    )

    if not download_and_verify_md5(
        session,
        f"https://www.cityscapes-dataset.com/md5-sum/?packageID={param['packageID']}",
        param["path"],
    ):
        console.log(f"{dataset} checksum verification failed.")


def download_and_verify_md5(session, url, file_path):
    console.log(f"Downloading MD5 checksum from {url}")

    r = session.get(url)
    r.raise_for_status()

    original_md5 = r.text.strip().split()[0]  # Taking only the first part of the string
    console.log(f"Original MD5 checksum: {original_md5}")

    # Calculate the MD5 checksum of the downloaded file
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        with Progress() as progress:
            task = progress.add_task(
                "[cyan]Calculating MD5 checksum...",
                total=Path(file_path).stat().st_size,
            )

            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
                progress.update(task, advance=len(byte_block))

    calculated_md5 = md5_hash.hexdigest()
    console.log(f"Calculated MD5 checksum: {calculated_md5}")

    if original_md5 == calculated_md5:
        console.log("[green]MD5 checksum verified.")
        console.log("")
        return True
    else:
        console.log("[red]MD5 checksum verification failed.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cityscapes dataset.")
    parser.add_argument(
        "--gtFine_zip", required=False, help="Path to gtFine_trainvaltest.zip"
    )
    parser.add_argument(
        "--leftImg8bit_zip", required=False, help="Path to leftImg8bit_trainvaltest.zip"
    )
    parser.add_argument("--gtCoarse_zip", required=False, help="Path to gtCoarse.zip")
    parser.add_argument(
        "--train_extra_zip", required=False, help="Path to leftImg8bit_trainextra.zip"
    )
    parser.add_argument(
        "--username", required=False, help="Cityscapes account username"
    )
    parser.add_argument(
        "--password", required=False, help="Cityscapes account password"
    )
    parser.add_argument(
        "--skip_md5",
        action="store_true",
        default=False,
        help="Skip MD5 checksum verification",
    )
    args = parser.parse_args()

    zip_data = {
        "gtFine": "../data/interim/gtFine_trainvaltest.zip",
        "leftImg8bit": "../data/interim/leftImg8bit_trainvaltest.zip",
        "gtCoarse": "../data/interim/gtCoarse.zip",
        "train_extra": "../data/interim/leftImg8bit_trainextra.zip",
    }

    for dataset, default_path in zip_data.items():
        user_zip_path_arg = getattr(args, f"{dataset}_zip", None)
        if user_zip_path_arg:
            user_zip_path = Path(user_zip_path_arg)
            if not user_zip_path.exists():
                console.log(f"{user_zip_path} does not exist.")
                exit(1)
            console.log(f"Using {user_zip_path} as {dataset}.zip")
        elif args.username and args.password:
            console.log(f"Downloading {dataset}.zip")
            login_and_download(args.username, args.password, dataset)
            setattr(args, f"{dataset}_zip", default_path)
        else:
            console.log(
                f"Please provide either the path to {dataset}.zip or your Cityscapes account credentials."
            )
            exit(1)

    package_ids = {"gtFine": 1, "leftImg8bit": 3, "gtCoarse": 2, "train_extra": 4}

    if not args.skip_md5 and not (args.username and args.password):
        console.log(
            "Skipping MD5 checksum verification because no Cityscapes account credentials were provided."
        )
    elif not args.skip_md5:
        session = requests.Session()
        login_data = {
            "username": args.username,
            "password": args.password,
            "submit": "Login",
        }
        r = session.post("https://www.cityscapes-dataset.com/login/", data=login_data)

        for dataset, package_id in package_ids.items():
            zip_path = getattr(args, f"{dataset}_zip", None)
            if zip_path:
                if not download_and_verify_md5(
                    session,
                    f"https://www.cityscapes-dataset.com/md5-sum/?packageID={package_id}",
                    zip_path,
                ):
                    console.log(f"{dataset} checksum verification failed.")
                    exit(1)

    temp_folder = Path("../data/interim/cityscapes")
    output_base_folder = Path("../data/processed/cityscapes")
    csv_base_path = Path("../data/processed/cityscapes/")

    for folder in [temp_folder, output_base_folder, csv_base_path]:
        folder.mkdir(parents=True, exist_ok=True)

    image_folder = temp_folder / "leftImg8bit"
    label_folder = temp_folder / "gtFine"
    coarse_folder = temp_folder / "gtCoarse"

    # If these folders exist already, we can skip unzipping
    if not (image_folder.exists() and label_folder.exists() and coarse_folder.exists()):
        for dataset in ["gtFine", "gtCoarse", "leftImg8bit", "train_extra"]:
            zip_file = getattr(args, f"{dataset}_zip", None)
            if zip_file:
                zip_path = Path(zip_file)
                unzip_files(zip_path, temp_folder)
            else:
                console.log(f"Skipping {dataset} because no zip file was provided.")
    else:
        print("Skipping unzipping because the folders already exist.")

    process_images(
        image_folder, label_folder, coarse_folder, output_base_folder, csv_base_path
    )

    # Join the train and train_extra CSVs
    train_csv = pd.read_csv(csv_base_path / "train_labels.csv")
    train_extra_csv = pd.read_csv(csv_base_path / "train_extra_labels.csv")
    train_csv = pd.concat([train_csv, train_extra_csv])
    train_csv.to_csv(csv_base_path / "train_labels.csv", index=False)

    # Remove the train_extra CSV
    (csv_base_path / "train_extra_labels.csv").unlink()

    # Move the train_extra images to the train folder
    for image in (output_base_folder / "train_extra").iterdir():
        image.rename(output_base_folder / "train" / image.name)

    # Remove the train_extra folder
    (output_base_folder / "train_extra").rmdir()

    print("Cityscapes dataset processing complete.")
