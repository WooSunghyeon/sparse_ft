"""RAPA fine-tuning pipeline integrations for LMFlow."""
from .sift_tuner import train_sift
from .spiel_tuner import train_spiel
from .smt_tuner import train_smt
from .s2ft_tuner import train_s2ft
from .ltsft_tuner import train_ltsft

METHODS = {
    "sift": train_sift,
    "spiel": train_spiel,
    "smt": train_smt,
    "s2ft": train_s2ft,
    "ltsft": train_ltsft,
}

__all__ = ["train_sift", "train_spiel", "train_smt", "train_s2ft", "train_ltsft", "METHODS"]
