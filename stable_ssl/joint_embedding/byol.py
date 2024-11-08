# -*- coding: utf-8 -*-
"""BYOL model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
import torch

from stable_ssl.utils import mlp
from .base import SelfDistillationModel, SelfDistillationConfig


class BYOL(SelfDistillationModel):
    """BYOL model from [GSA+20].

    Reference
    ---------
    .. [GSA+20] Grill, J. B., Strub, F., Altché, ... & Valko, M. (2020).
            Bootstrap Your Own Latent-A New Approach To Self-Supervised Learning.
            Advances in neural information processing systems, 33, 21271-21284.
    """

    def initialize_modules(self):
        super().initialize_modules()

        sizes = [self.config.model.projector[-1]] + self.config.model.predictor
        self.predictor = mlp(sizes)

    def compute_ssl_loss(self, z_0, z_1):
        """Compute the loss of the BYOL model.

        Parameters
        ----------
        z_0 : torch.Tensor
            Latent representation of the first augmented view of the batch.
        z_1 : torch.Tensor
            Latent representation of the second augmented view of the batch.

        Returns
        -------
        float
            The computed loss.
        """
        # BYOL relies on a predictor network on top of the projector.
        prediction_0 = self.predictor(z_0)
        prediction_1 = self.predictor(z_1)

        # Construct target with the target ('teacher') network.
        with torch.no_grad():
            projection_target_0 = self.projector_target(
                self.backbone_target(self.data[0][0])
            )
            projection_target_1 = self.projector_target(
                self.backbone_target(self.data[0][1])
            )

        sim = torch.nn.CosineSimilarity(dim=1)
        return -0.5 * (
            sim(prediction_0, projection_target_1).mean()
            + sim(prediction_1, projection_target_0).mean()
        )


@dataclass
class BYOLConfig(SelfDistillationConfig):
    """Configuration for the BYOL model parameters.

    Parameters
    ----------
    predictor : str
        Architecture of the predictor head. Default is "2048-256".
    """

    predictor: list[int] = field(default_factory=lambda: [2048, 256])

    def __post_init__(self):
        """Convert predictor string to a list of integers if necessary."""
        super().__post_init__()
        if isinstance(self.predictor, str):
            self.predictor = [int(i) for i in self.predictor.split("-")]

    def trainer(self):
        return BYOL
