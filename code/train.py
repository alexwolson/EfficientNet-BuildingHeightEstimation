# Standard Libraries
from pathlib import Path
import argparse

from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

# PyTorch and PyTorch Lightning
import torch
import pytorch_lightning as pl
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Optuna
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# torchvision
from torchvision import transforms

# Custom Modules
from building_dataset import BuildingDataset
from building_model import BuildingModelSm, BuildingModelMed, BuildingModelLg

# On modern GPUs, we can speed up training by reducing the precision of the matrix multiplication operations.
torch.set_float32_matmul_precision("medium")

def get_mean_std(loader: DataLoader) -> tuple[float, float]:
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader, total=len(loader), desc="Calculating Mean and Std"):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def build_transforms(
    train_dataset: BuildingDataset,
) -> tuple[transforms.Compose, transforms.Compose]:
    loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    mean, std = get_mean_std(loader)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(640),
            transforms.Resize((660, 660)),
            transforms.RandomCrop(640),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.CenterCrop(640),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform


def visualize_transformed_images(
    dataset: BuildingDataset,
    transform: transforms.Compose,
    num_images: int = 5,
    save_path: Path = None,
) -> None:
    import matplotlib.pyplot as plt

    for i in range(num_images):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        image, label = dataset[i]
        numpy_image = image.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()

        # Show original image
        axes[0].imshow(numpy_image)
        axes[0].set_title(f"Original Image\nLabel: {label}")

        # Convert to PIL Image for transformation
        pil_image = Image.fromarray((numpy_image * 255).astype("uint8"), "RGB")

        # Apply all transformations
        transformed_image = transform(pil_image)

        # Convert transformed image back to numpy array for visualization
        transformed_numpy = (
            transformed_image.permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
        )

        # Show transformed image
        axes[1].imshow(transformed_numpy)
        axes[1].set_title("Transformed Image")

        if save_path:
            filename = save_path / f"image_{i + 1}_label_{label}.png"
            plt.savefig(filename)

        plt.close(fig)


def objective(trial: optuna.Trial) -> float:
    activation = trial.suggest_categorical(
        "activation", ["relu", "leaky_relu", "selu", "swish"]
    )
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    loss = trial.suggest_categorical("loss", ["mse", "l1"])
    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "lion"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    print(f"Building model with parameters: {trial.params}")

    model = BuildingModelSm(
        activation=activation,
        dropout=dropout,
        loss=loss,
        optimizer=optimizer,
        optimizer_params={"lr": lr, "weight_decay": weight_decay},
    )

    # Initialize ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_mae"),
            checkpoint_callback,
        ],
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)

    print(f"Finished training with parameters: {trial.params}")
    print(f"Metrics: {trainer.logged_metrics}")

    # Return the best validation MAE
    return checkpoint_callback.best_model_score.item()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--batch_size", type=int, default=8)
    argparser.add_argument("--epochs", type=int, default=100)
    argparser.add_argument("--checkpoint", type=str, default=None)
    argparser.add_argument(
        "--data_path", type=str, default="../data/processed/massings"
    )
    argparser.add_argument("--experiment_name", type=str, default="default")
    argparser.add_argument("--visualize", action="store_true", default=False)
    argparser.add_argument("--predictor", type=str, default="LOG_AVG_HEIGHT")
    argparser.add_argument(
        "--optuna",
        action="store_true",
        default=False,
        help="Run optuna hyperparameter search",
    )

    args = argparser.parse_args()

    if "LOG" not in args.predictor:
        model = BuildingModelSm(
            activation="relu",
            dropout=0.1006336947864703,
            loss="l1",
            optimizer="lion",
            optimizer_params={
                "lr": 2.6346031838180948e-05,
                "weight_decay": 5.736285013612257e-05,
            },
        )
    else:
        model = BuildingModelSm(
            activation="swish",
            dropout=0.13879735793042342,
            loss="l1",
            optimizer="lion",
            optimizer_params={
                "lr": 1.7252597213845926e-05,
                "weight_decay": 0.00011943689536244606,
            },
        )

    print("Built model: ", model)

    if args.checkpoint:
        if not Path(args.checkpoint).exists():
            raise ValueError(f"Checkpoint {args.checkpoint} does not exist.")
        model = model.load_from_checkpoint(args.checkpoint)

    if not Path(args.data_path).exists():
        raise ValueError(f"Data path {args.data_path} does not exist.")
    else:
        data_path = Path(args.data_path)

    train_transform_path = Path(
        f"../data/interim/train_transform_{args.experiment_name}.pt"
    )
    val_transform_path = Path(
        f"../data/interim/val_transform_{args.experiment_name}.pt"
    )

    if (
        train_transform_path.exists()
        and val_transform_path.exists()
        and not args.visualize
    ):
        print("Loading Transforms")
        train_transform = torch.load(train_transform_path)
        val_transform = torch.load(val_transform_path)
    else:
        print("Building Transforms")
        train_data = BuildingDataset(
            root_dir=data_path / "train",
            csv_file=data_path / "train_labels.csv",
            label_column=args.predictor,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
        )

        train_transform, val_transform = build_transforms(train_data)
        torch.save(train_transform, train_transform_path)
        torch.save(val_transform, val_transform_path)

        if args.visualize:
            visualize_transformed_images(
                train_data,
                train_transform,
                num_images=10,
                save_path=Path(f"../reports/train_transform_{args.experiment_name}"),
            )

    print("Building Datasets")

    train_data = BuildingDataset(
        root_dir=data_path / "train",
        csv_file=data_path / "train_labels.csv",
        label_column=args.predictor,
        transform=train_transform,
    )

    val_data = BuildingDataset(
        root_dir=data_path / "val",
        csv_file=data_path / "val_labels.csv",
        label_column=args.predictor,
        transform=val_transform,
    )

    print("Preparing Dataloaders")

    BATCH_SIZE = args.batch_size

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=12
    )

    if args.optuna:
        study = optuna.create_study(
            direction="minimize", study_name=args.experiment_name
        )
        study.optimize(objective, n_trials=50)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        exit()

    # Model and Trainer
    early_stop_callback = EarlyStopping(
        monitor="val_mae", min_delta=0.00, patience=10, verbose=True, mode="min"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae",
        dirpath="../models",
        filename=f"{args.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save and record metrics
    metrics = trainer.logged_metrics
    metrics["batch_size"] = args.batch_size
    metrics["epochs"] = args.epochs

    with open(
        f"../reports/metrics_{args.experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
        "a",
    ) as f:
        f.write(",".join(metrics.keys()) + "\n")
        f.write(",".join([str(v) for v in metrics.values()]) + "\n")
