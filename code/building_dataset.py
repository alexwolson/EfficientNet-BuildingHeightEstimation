from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import torch


class BuildingDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        csv_file: Path,
        label_column: str,
        transform: "torchvision.transforms.Compose",
    ) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_name = self.root_dir / f"{self.df.iloc[idx, 0]}.jpg"
        image = Image.open(img_name)
        label = self.df.loc[idx, self.label_column]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __repr__(self) -> str:
        return f"BuildingDataset(root_dir={self.root_dir}, label_column={self.label_column}, transform={self.transform})"
