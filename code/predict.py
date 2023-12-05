from building_model import BuildingModelSm
from building_dataset import BuildingDataset
import argparse
import torch
from torch.utils.data import DataLoader
from rich.console import Console
from pathlib import Path

console = Console()


def predict(model, data_loader, device):
    model.eval()
    yhat = []
    for batch in data_loader:
        batch = tuple(t.to(device) for t in batch)
        inputs, _ = batch
        with torch.no_grad():
            outputs = model(inputs)
            yhat.append(outputs)
    return yhat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="../reports")
    parser.add_argument("--label", type=str, default="AVG_HEIGHT")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        console.log("Using CUDA")
    else:
        console.log("Using CPU")

    if not args.checkpoint.exists():
        console.log(f"{args.checkpoint} does not exist")
        exit(1)

    if not args.dataset.exists():
        console.log(f"{args.dataset} does not exist")
        exit(1)

    val_transform_path = Path(f"../data/interim/val_transform_{args.experiment}.pt")

    if not val_transform_path.exists():
        console.log(f"{val_transform_path} does not exist")
        exit(1)

    model = BuildingModelSm.load_from_checkpoint(args.checkpoint)

    dataset = BuildingDataset(
        root_dir=args.dataset / "test",
        csv_file=args.dataset / "test_labels.csv",
        label_column=args.label,
        transform=torch.load(val_transform_path),
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    preds = predict(model, dataloader, device)

    # Report MAE, std dev, and % below 2m
    y = []
    for i in range(len(dataset)):
        y.append(dataset[i][1].item())
    y = torch.tensor(y)
    yhat = torch.tensor([item for sublist in preds for item in sublist.cpu().numpy()])
    mae = torch.mean(torch.abs(y - yhat))
    console.log(f"MAE: {mae}")
    console.log(f"STD: {torch.std(y - yhat)}")
    console.log(
        f"% below 2m: {torch.sum(torch.abs(yhat - y) < 2) / torch.tensor(len(y), dtype=torch.float)}"
    )

    # Save predictions
    with open(args.output / f"preds_{args.experiment}.csv", "w") as f:
        f.write("id,AVG_HEIGHT,PREDICTED\n")
        for i in range(len(dataset)):
            f.write(f"{dataset.df.iloc[i, 0]}, {dataset.df.iloc[i, 3]}, {yhat[i].item()}\n")

    print("Wrote predictions to", args.output / f"preds_{args.experiment}.csv")
