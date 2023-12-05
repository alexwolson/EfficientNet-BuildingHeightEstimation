# Building Height Prediction

This repository contains a Python-based machine learning project for predicting building heights. The project uses PyTorch, PyTorch Lightning, Optuna, and Google Street View API for data downloading, processing, model training, and prediction.

## Project Structure

The project is structured as follows:

- `download_massings.py`: Downloads a dataset of 3D building massings from the Toronto Open Data portal, unzips the data, and processes it into a pickle file.
- `download_streetview.py`: Uses the Google Street View API to download images of buildings. It keeps track of which buildings have been photographed and saves the images to a specified directory.
- `download_cityscapes.py`: Downloads and processes the Cityscapes dataset, a popular dataset for semantic urban scene understanding tasks in computer vision.
- `predict.py`: Loads a trained model and a dataset, makes predictions on the dataset, and saves the predictions to a CSV file. It also reports some metrics about the predictions, such as Mean Absolute Error (MAE) and standard deviation.
- `train.py`: The main training script. It loads the data, applies transformations, and trains a model using PyTorch Lightning. It also supports hyperparameter optimization using Optuna.
- `building_dataset.py`: Defines the `BuildingDataset` class used in the training and prediction scripts.
- `building_model.py`: Defines the `BuildingModelSm` class used in the training and prediction scripts.

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- Optuna
- Google Street View API

## Usage

1. Clone the repository.
2. Install the required packages.
3. Run the `download_massings.py`, `download_streetview.py`, and `download_cityscapes.py` scripts to download and process the data.
4. Run the `train.py` script to train the model.
5. Run the `predict.py` script to make predictions on the dataset.
