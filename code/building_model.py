import torch
from torchvision.models import (
    efficientnet_v2_s,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
)
import pytorch_lightning as pl
import torch.nn.functional as F
import optuna
from lion_pytorch import Lion


def below_threshold(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 2,
    exponentiate: bool = True,
) -> torch.Tensor:
    if exponentiate:
        y_hat = torch.exp(y_hat)
        y = torch.exp(y)
    return torch.sum(torch.abs(y_hat - y) < threshold) / torch.tensor(
        len(y), dtype=torch.float
    )


class BuildingModel(pl.LightningModule):
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat.squeeze(), y.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        y_hat = self.forward(x)
        # Calculate two versions of each metric: one for if it's not log transformed, and one for if it is
        val_mae = self.val_mae_metric(y_hat.squeeze(), y.float())
        val_mae_exp = self.val_mae_metric(torch.exp(y_hat.squeeze()), torch.exp(y))
        val_below_threshold = below_threshold(
            y_hat.squeeze(), y.float(), exponentiate=False
        )
        val_below_threshold_exp = below_threshold(
            y_hat.squeeze(), y.float(), exponentiate=True
        )
        self.log("val_mae", val_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_mae_exp", val_mae_exp, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_below_2m",
            val_below_threshold,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_below_2m_exp",
            val_below_threshold_exp,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self) -> dict[str, object]:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_mae",
        }


class BuildingModelSm(BuildingModel):
    def __init__(
        self,
        activation: str = "swish",
        dropout: float = 0.13879735793042342,
        loss: str = "l1",
        optimizer: str = "lion",
        optimizer_params: dict[str, object] = {
            "lr": 1.7252597213845926e-05,
            "weight_decay": 0.00011943689536244606,
        },
    ) -> None:
        super().__init__()
        match activation:
            case "relu":
                activation = torch.nn.ReLU
            case "leaky_relu":
                activation = torch.nn.LeakyReLU
            case "selu":
                activation = torch.nn.SELU
            case "swish":
                activation = torch.nn.SiLU
            case _:
                raise ValueError(f"Activation {activation} not supported")
        match loss:
            case "mse":
                loss = F.mse_loss
            case "l1":
                loss = F.l1_loss
            case _:
                raise ValueError(f"Loss {loss} not supported")
        match optimizer:
            case "adam":
                optimizer = torch.optim.Adam
            case "adamw":
                optimizer = torch.optim.AdamW
            case "lion":
                optimizer = Lion
            case _:
                raise ValueError(f"Optimizer {optimizer} not supported")

        self.model = efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(500, 250),
            activation(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(250, 1),
        )
        self.val_mae_metric = F.l1_loss
        self.loss = loss
        self.optimizer = optimizer(self.parameters(), **optimizer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.model(x)).squeeze(-1)

    def __repr__(self) -> str:
        return f"BuildingModelSm(activation={self.regressor[1]}, dropout={self.regressor[2].p}, loss={self.loss}, optimizer={self.optimizer.__class__.__name__}, optimizer_params={self.optimizer.defaults})"


class BuildingModelMed(BuildingModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 1),
        )
        self.val_mae_metric = F.l1_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.model(x)).squeeze(-1)


class BuildingModelLg(BuildingModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 1),
        )
        self.val_mae_metric = F.l1_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.model(x)).squeeze(-1)
