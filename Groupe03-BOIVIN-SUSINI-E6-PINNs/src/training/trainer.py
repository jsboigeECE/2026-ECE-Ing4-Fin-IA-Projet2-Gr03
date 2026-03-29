"""
PINNTrainer — generic training loop for PINN models.

Features
--------
- Adam optimiser with optional L-BFGS fine-tuning phase
- Periodic checkpoint saving (every `save_every` epochs)
- TensorBoard logging (optional)
- Early stopping on total loss
- Clean tqdm progress bar
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import LossDecomposition


class PINNTrainer:
    """
    Parameters
    ----------
    model        : a BlackScholesPINN (or any model with .compute_loss())
    n_epochs     : number of Adam training epochs
    lr           : Adam learning rate
    n_coll       : collocation points per epoch
    n_bc         : boundary / IC points per epoch
    lbfgs_epochs : optional BFGS fine-tuning epochs after Adam (0 = disabled)
    save_every   : checkpoint interval (0 = no checkpoints)
    checkpoint_dir : where to save .pt checkpoints
    patience     : early-stopping patience (epochs without improvement, 0 = off)
    device       : torch.device (defaults to CUDA if available, else CPU)
    """

    def __init__(
        self,
        model,
        n_epochs: int = 10_000,
        lr: float = 1e-3,
        n_coll: int = 5_000,
        n_bc: int = 500,
        lbfgs_epochs: int = 0,
        save_every: int = 1_000,
        checkpoint_dir: str = "experiments/results/models",
        patience: int = 500,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.n_coll = n_coll
        self.n_bc = n_bc
        self.lbfgs_epochs = lbfgs_epochs
        self.save_every = save_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.patience = patience
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.history = LossDecomposition()

    # ------------------------------------------------------------------

    def train(self) -> LossDecomposition:
        """Run the full training procedure and return loss history."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=200, min_lr=1e-6
        )

        best_loss = float("inf")
        no_improve = 0
        t0 = time.time()

        pbar = tqdm(range(1, self.n_epochs + 1), desc="Training", ncols=90)
        for epoch in pbar:
            self.model.train()
            optimizer.zero_grad()

            loss_dict = self.model.compute_loss(
                n_coll=self.n_coll, n_bc=self.n_bc, device=self.device
            )
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(loss_dict["total"].detach())

            self.history.update(loss_dict)
            cur = loss_dict["total"].item()

            # progress bar
            pbar.set_postfix(
                loss=f"{cur:.4e}",
                pde=f"{loss_dict['pde'].item():.3e}",
                bc=f"{loss_dict['bc'].item():.3e}",
                ic=f"{loss_dict['ic'].item():.3e}",
            )

            # early stopping
            if cur < best_loss - 1e-8:
                best_loss = cur
                no_improve = 0
            else:
                no_improve += 1
            if self.patience > 0 and no_improve >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {self.patience} epochs)")
                break

            # checkpoint
            if self.save_every > 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

        elapsed = time.time() - t0
        print(f"\nTraining complete in {elapsed:.1f}s — best loss: {best_loss:.4e}")

        # optional L-BFGS fine-tuning
        if self.lbfgs_epochs > 0:
            self._lbfgs_phase()

        return self.history

    # ------------------------------------------------------------------

    def _lbfgs_phase(self):
        print(f"Starting L-BFGS fine-tuning for up to {self.lbfgs_epochs} steps...")
        optimizer = optim.LBFGS(
            self.model.parameters(),
            max_iter=self.lbfgs_epochs,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            ld = self.model.compute_loss(
                n_coll=self.n_coll, n_bc=self.n_bc, device=self.device
            )
            ld["total"].backward()
            self.history.update(ld)
            return ld["total"]

        optimizer.step(closure)
        print(f"L-BFGS done — final loss: {self.history.total[-1]:.4e}")

    def _save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"pinn_bs_epoch{epoch:05d}.pt"
        torch.save(
            {"epoch": epoch, "model_state": self.model.state_dict(),
             "history": self.history},
            path,
        )

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", LossDecomposition())
        print(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
