"""Latent-variable fusion of GRACE-FO and Sentinel-2 retrievals."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionConfig:
    latent_dim: int = 8
    iterations: int = 500


class LatentFusionModel:
    def __init__(self, config: FusionConfig, seed: int = 0):
        self.config = config
        rng = np.random.default_rng(seed)
        self.encoder_a = rng.normal(size=(4, config.latent_dim))
        self.encoder_b = rng.normal(size=(6, config.latent_dim))
        self.decoder = rng.normal(size=(config.latent_dim, 1))

    def encode(self, grace_features: np.ndarray, sentinel_features: np.ndarray) -> np.ndarray:
        a = np.tanh(grace_features @ self.encoder_a)
        b = np.tanh(sentinel_features @ self.encoder_b)
        return (a + b) / 2.0

    def recharge(self, grace_features: np.ndarray, sentinel_features: np.ndarray) -> np.ndarray:
        latent = self.encode(grace_features, sentinel_features)
        return np.maximum(latent @ self.decoder, 0.0).ravel()
