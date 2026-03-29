"""
LossDecomposition: thin wrapper that stores per-component loss histories.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import torch


@dataclass
class LossDecomposition:
    """Accumulates loss values across epochs for later plotting."""

    total:     List[float] = field(default_factory=list)
    pde:       List[float] = field(default_factory=list)
    bc:        List[float] = field(default_factory=list)
    ic:        List[float] = field(default_factory=list)
    penalty:   List[float] = field(default_factory=list)  # American put penalty

    def update(self, loss_dict: Dict[str, torch.Tensor]):
        self.total.append(loss_dict["total"].item())
        self.pde.append(loss_dict["pde"].item())
        self.bc.append(loss_dict["bc"].item())
        self.ic.append(loss_dict["ic"].item())
        if "penalty" in loss_dict:
            self.penalty.append(loss_dict["penalty"].item())

    def last(self) -> Dict[str, float]:
        d = {
            "total": self.total[-1],
            "pde":   self.pde[-1],
            "bc":    self.bc[-1],
            "ic":    self.ic[-1],
        }
        if self.penalty:
            d["penalty"] = self.penalty[-1]
        return d
