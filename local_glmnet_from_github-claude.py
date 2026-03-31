"""
LocalGLMnet — PyTorch Lightning Implementation
================================================
Berdasarkan arsitektur dari Kode R yang menggunakan Keras/TensorFlow.
Sumber: https://github.com/salvatorescognamiglio/mortality_forecasting_localGLMnet/tree/main#

Arsitektur:
    Input  : (batch, 10, 100)  — mortality rates 10 tahun, 100 kelompok usia
    Output : (batch, 2, 100)   — [forecast_rates, penalty]

Pipeline:
    1. Attention Layer  (LocallyConnected2D + Sigmoid) → interim
    2. Masked Decode    (rates * interim)               → decoded_masked
    3. Forecast         (sum atas dimensi lag)          → forecast_rates (batch, 1, 100)
    4. Elastic Net Penalty                              → penalty        (batch, 1, 100)
    5. Concat           ([forecast_rates, penalty])     → output         (batch, 2, 100)

Loss:
    MSEP = mean( (y_true[:,0,:] - y_pred[:,0,:])^2 + y_pred[:,1,:] )
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

# ─────────────────────────────────────────────────────────────
# 1. Dataset
# ─────────────────────────────────────────────────────────────


class MortalityDataset(Dataset):
    """
    Dataset untuk mortality rates.

    Args:
        x : tensor shape (n_sample, 10, 100) — mortality rates masa lalu
        y : tensor shape (n_sample, 1, 100)  — mortality rates target
        ids: list of string metadata per sample (opsional)
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, ids: list | None = None):
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Jumlah sample x ({x.shape[0]}) dan y ({y.shape[0]}) harus sama"
            )
        self.x = x.float()
        self.y = y.float()
        self.ids = ids

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.ids is not None:
            return self.x[idx], self.y[idx], self.ids[idx]
        return self.x[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        """Custom collate untuk handle metadata string."""
        if len(batch[0]) == 3:
            xs, ys, ids = zip(*batch)
            return torch.stack(xs), torch.stack(ys), list(ids)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


# ─────────────────────────────────────────────────────────────
# 2. LocallyConnected2D
# ─────────────────────────────────────────────────────────────


class LocallyConnected2d(nn.Module):
    """
    Locally Connected 2D Layer — seperti Conv2D tapi TANPA weight sharing.
    Setiap posisi spasial punya kernel sendiri.

    Equivalent dengan layer_locally_connected_2d di Keras.

    Args:
        in_channels  : jumlah input channel
        out_channels : jumlah output channel (filter)
        input_size   : (H, W) ukuran input setelah padding
        kernel_size  : ukuran kernel (kH, kW)
        activation   : activation function (default: sigmoid)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int],
        kernel_size: tuple[int, int],
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation if activation is not None else nn.Sigmoid()

        H_in, W_in = input_size
        kH, kW = kernel_size

        # Output size (tanpa padding, stride=1)
        self.H_out = H_in - kH + 1
        self.W_out = W_in - kW + 1

        # Tiap posisi output punya kernel sendiri
        # weight shape: (H_out, W_out, out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(self.H_out, self.W_out, out_channels, in_channels, kH, kW)
            * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, in_channels, H, W)
        batch = x.shape[0]
        kH, kW = self.kernel_size
        out = torch.zeros(
            batch,
            self.out_channels,
            self.H_out,
            self.W_out,
            device=x.device,
            dtype=x.dtype,
        )

        for i in range(self.H_out):
            for j in range(self.W_out):
                # Patch: (batch, in_channels, kH, kW)
                patch = x[:, :, i : i + kH, j : j + kW]
                # weight[i,j]: (out_channels, in_channels, kH, kW)
                w = self.weight[i, j]
                # Einsum: (batch, out_channels)
                out[:, :, i, j] = torch.einsum("bckl,ockl->bo", patch, w)

        return self.activation(out)


# ─────────────────────────────────────────────────────────────
# 3. LocalGLMnet nn.Module
# ─────────────────────────────────────────────────────────────


class LocalGLMnet(nn.Module):
    """
    LocalGLMnet architecture.

    Args:
        look_back  : jumlah tahun ke belakang (default: 10)
        n_ages     : jumlah kelompok usia     (default: 100)
        eta        : regularization strength  (default: 0)
        alpha      : elastic net mixing       (default: 0, 0=Ridge, 1=Lasso)
        pad        : padding size untuk locally connected (default: 2)
        kernel_size: ukuran kernel locally connected (default: (5,5))
    """

    def __init__(
        self,
        look_back: int = 10,
        n_ages: int = 100,
        eta: float = 0.0,
        alpha: float = 0.0,
        pad: int = 2,
        kernel_size: tuple[int, int] = (5, 5),
    ):
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha harus antara 0 dan 1, dapat {alpha}")
        if eta < 0:
            raise ValueError(f"eta harus >= 0, dapat {eta}")

        self.look_back = look_back
        self.n_ages = n_ages
        self.eta = eta
        self.alpha = alpha
        self.pad = pad
        self.kernel_size = kernel_size

        # Input size setelah padding: (look_back + 2*pad, n_ages + 2*pad)
        H_padded = look_back + 2 * pad  # 10 + 4 = 14
        W_padded = n_ages + 2 * pad  # 100 + 4 = 104

        # ── Attention Layer ──────────────────────────────────
        # LocallyConnected2D: in_channels=1, out_channels=1
        # kernel=(5,5) → output size = (14-5+1, 104-5+1) = (10, 100) ✓
        self.locally_connected = LocallyConnected2d(
            in_channels=1,
            out_channels=1,
            input_size=(H_padded, W_padded),
            kernel_size=kernel_size,
            activation=nn.Sigmoid(),
        )

        # ── Forecast Layer ───────────────────────────────────
        # TimeDistributed Dense(1, no bias, weights=ones, frozen)
        # Equivalent dengan sum atas dimensi lag → tidak perlu layer, cukup .sum()
        # Tapi untuk faithful reproduction, kita buat sebagai parameter frozen
        self.forecast_weight = nn.Parameter(
            torch.ones(look_back, 1), requires_grad=False  # frozen, persis seperti di R
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, look_back, n_ages)
        Returns:
            output: (batch, 2, n_ages) — [forecast_rates, penalty]
        """
        # ── Step 1: Attention ────────────────────────────────
        # (batch, 10, 100) → (batch, 1, 10, 100)
        x_4d = x.unsqueeze(1)

        # ZeroPad2D: (batch, 1, 10+2*pad, 100+2*pad)
        x_padded = F.pad(x_4d, (self.pad, self.pad, self.pad, self.pad))

        # LocallyConnected2D + sigmoid: (batch, 1, 10, 100)
        interim_4d = self.locally_connected(x_padded)

        # (batch, 1, 10, 100) → (batch, 10, 100)
        interim = interim_4d.squeeze(1)

        # ── Step 2: Masked Decode ────────────────────────────
        # Element-wise multiply: (batch, 10, 100) * (batch, 10, 100)
        decoded_masked = x * interim

        # ── Step 3: Forecast ─────────────────────────────────
        # Sum atas dimensi lag (dim=1): (batch, 10, 100) → (batch, 1, 100)
        # Equivalent dengan TimeDistributed Dense(weights=ones, no bias)
        forecast_rates = decoded_masked.sum(dim=1, keepdim=True)

        # ── Step 4: Elastic Net Penalty ───────────────────────
        # L1: eta * alpha * sqrt(interim^2 + eps) summed over lag
        # L2: eta * (1-alpha) * interim^2 summed over lag
        eps = 1e-6
        l1_penalty = self.eta * self.alpha * torch.sqrt(interim**2 + eps)
        l2_penalty = self.eta * (1 - self.alpha) * interim**2

        # Sum atas dimensi lag: (batch, 10, 100) → (batch, 1, 100)
        penalty = (l1_penalty + l2_penalty).sum(dim=1, keepdim=True)

        # ── Step 5: Concat ────────────────────────────────────
        # (batch, 2, 100)
        output = torch.cat([forecast_rates, penalty], dim=1)

        return output

    def get_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Untuk analisis — return interim dan decoded_masked.
        Equivalent dengan model_attention di kode R.

        Returns:
            interim       : (batch, 10, 100) — attention coefficients
            decoded_masked: (batch, 10, 100) — contribution values
        """
        x_padded = F.pad(x.unsqueeze(1), (self.pad, self.pad, self.pad, self.pad))
        interim = self.locally_connected(x_padded).squeeze(1)
        decoded = x * interim
        return interim, decoded


# ─────────────────────────────────────────────────────────────
# 4. PyTorch Lightning Module
# ─────────────────────────────────────────────────────────────


class LocalGLMnetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper untuk LocalGLMnet.

    Args:
        look_back   : jumlah tahun ke belakang
        n_ages      : jumlah kelompok usia
        eta         : regularization strength
        alpha       : elastic net mixing (0=Ridge, 1=Lasso)
        lr          : learning rate awal
        lr_factor   : faktor reduksi LR saat plateau
        lr_patience : patience untuk ReduceLROnPlateau
        lr_min      : minimum learning rate
        lr_cooldown : cooldown ReduceLROnPlateau
    """

    def __init__(
        self,
        look_back: int = 10,
        n_ages: int = 100,
        eta: float = 0.0,
        alpha: float = 0.0,
        lr: float = 1e-3,
        lr_factor: float = 0.90,
        lr_patience: int = 50,
        lr_min: float = 5e-5,
        lr_cooldown: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LocalGLMnet(
            look_back=look_back,
            n_ages=n_ages,
            eta=eta,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _msep_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        MSEP Regularized Loss — persis seperti di kode R:
            mean( (y_true[:,0,:] - y_pred[:,0,:])^2 + y_pred[:,1,:] )
        """
        mse = (y_true[:, 0, :] - y_pred[:, 0, :]) ** 2
        penalty = y_pred[:, 1, :]
        return (mse + penalty).mean()

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred = self(x)
        loss = self._msep_loss(y_pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_pred = self(x)
        loss = self._msep_loss(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # NAdam — equivalent dengan optimizer_nadam() di R
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.hparams.lr)

        # ReduceLROnPlateau — equivalent dengan callback_reduce_lr_on_plateau di R
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
            cooldown=self.hparams.lr_cooldown,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # pantau val_loss untuk plateau
            },
        }

    def predict_attention(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention coefficients dan contribution values."""
        self.eval()
        with torch.no_grad():
            return self.model.get_attention(x)


# ─────────────────────────────────────────────────────────────
# 5. Training Script
# ─────────────────────────────────────────────────────────────


def train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    ids: list | None = None,
    eta: float = 0.0,
    alpha: float = 0.0,
    n_epochs: int = 500,
    batch_size: int = 16,
    val_split: float = 0.05,
    checkpoint_dir: str = "checkpoints",
    device: str = "auto",
) -> LocalGLMnetLightning:
    """
    Melatih LocalGLMnet.

    Args:
        x_train       : tensor (n_sample, 10, 100)
        y_train       : tensor (n_sample, 1, 100)
        ids           : metadata per sample (opsional)
        eta           : regularization strength
        alpha         : elastic net mixing
        n_epochs      : jumlah epoch
        batch_size    : batch size
        val_split     : proporsi validation set
        checkpoint_dir: direktori simpan checkpoint
        device        : "auto", "cpu", "cuda", "mps"

    Returns:
        model yang sudah dilatih dengan best weights
    """
    # ── Dataset & DataLoader ─────────────────────────────────
    dataset = MortalityDataset(x_train, y_train, ids)
    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=MortalityDataset.collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=MortalityDataset.collate_fn,
    )

    # ── Model ────────────────────────────────────────────────
    model = LocalGLMnetLightning(eta=eta, alpha=alpha)

    # ── Callbacks ────────────────────────────────────────────
    # ModelCheckpoint — equivalent dengan callback_model_checkpoint di R
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=True,
    )

    # ── Trainer ──────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator=device,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dl, val_dl)

    # Load best weights setelah training selesai
    best_model = LocalGLMnetLightning.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )

    return best_model


# ─────────────────────────────────────────────────────────────
# 6. Contoh Pemakaian
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)

    # Simulasi data — ganti dengan data asli
    n_sample = 200
    look_back = 10
    n_ages = 100

    x_train = torch.rand(n_sample, look_back, n_ages) * 0.1  # mortality rates kecil
    y_train = torch.rand(n_sample, 1, n_ages) * 0.1

    # Training
    trained_model = train(
        x_train=x_train,
        y_train=y_train,
        eta=0.0,  # ganti sesuai kebutuhan
        alpha=0.0,  # 0=Ridge, 1=Lasso
        n_epochs=10,  # ganti ke 500 untuk training penuh
        batch_size=16,
        val_split=0.05,
        checkpoint_dir="localGLMnet_checkpoints",
    )

    # Analisis attention coefficients
    interim, decoded_masked = trained_model.predict_attention(x_train)

    print(f"Attention coefficients shape : {interim.shape}")  # (200, 10, 100)
    print(f"Contribution values shape    : {decoded_masked.shape}")  # (200, 10, 100)

    # Untuk plot — equivalent dengan ggplot di kode R
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # for samp in range(n_sample):
    #     df = pd.DataFrame({
    #         "value"  : interim[samp].flatten().numpy(),
    #         "decoded": decoded_masked[samp].flatten().numpy(),
    #         "mx"     : x_train[samp].flatten().numpy(),
    #         "age"    : [age for _ in range(look_back) for age in range(n_ages)],
    #         "lag"    : [look_back - lag for lag in range(look_back) for _ in range(n_ages)],
    #     })
