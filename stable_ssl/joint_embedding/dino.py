# -*- coding: utf-8 -*-
"""DINO model."""
#
# Author: Hugues Van Assel <vanasselhugues@gmail.com>
#         Randall Balestriero <randallbalestriero@gmail.com>
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from stable_ssl.utils import log_and_raise
from .base import SelfDistillationConfig, SelfDistillationModel


class DINO(SelfDistillationModel):
    """DINO model from [CTM+21]_.

    Reference
    ---------
    .. [CTM+21] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J.,
            Bojanowski, P., & Joulin, A. (2021).
            Emerging Properties in Self-Supervised Vision Transformers.
            International Conference on Computer Vision.
    """

    def before_train_all_epochs(self):
        """Initialize the teacher temperature schedule."""
        self.teacher_temperature_schedule = torch.linspace(
            start=self.config.model.warmup_teacher_temperature,
            end=self.config.model.teacher_temperature,
            steps=self.config.model.warmup_teacher_temperature_epochs,
        )

    def compute_ssl_loss(self, *projections):
        """Compute the loss of the DINO model.

        Parameters
        ----------
        projections : List[torch.Tensor]
            List of the student's projections of the input views.

        Returns
        -------
        float
            The computed loss.
        """

        if len(projections) <= 2:
            log_and_raise(
                ValueError, 
                "DINO requires strictly more than 2 views. "
                "The first two views are global views."
            )

        student_out = torch.stack(projections)
        s_out = F.log_softmax(student_out / self.config.model.student_temperature, dim=-1)

        # Construct target *from global views only* with the target ('teacher') network.
        with torch.no_grad():
            global_views = self.data[0][:2]  # First two views are global views.
            projections_target = [
                self.projector_target(self.backbone_target(view))
                for view in global_views
            ]

        if self.epoch < self.warmup_teacher_temperature_epochs:
            teacher_temp = self.teacher_temperature_schedule[self.epoch]
        else:
            teacher_temp = self.config.model.teacher_temperature

        teacher_out = torch.stack(projections_target)
        t_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)

            

            # Calculate feature similarities, ignoring the diagonal
            # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
            loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
            loss.fill_diagonal_(0)

            # Number of loss terms, ignoring the diagonal
            n_terms = loss.numel() - loss.diagonal().numel()
            batch_size = teacher_out.shape[1]

            loss = loss.sum() / (n_terms * batch_size)

            # Update the center used for the teacher output
            self.update_center(teacher_out)

            return loss

    @torch.no_grad()
    def after_eval_step(self):
        # Calculate the batch center using the specified center function
        batch_center = self._center_fn(x=teacher_out, dim=(0, 1))

        # Update the center with a moving average
        self.center = center.center_momentum(
            center=self.center, batch_center=batch_center, momentum=self.center_momentum
        )


@torch.no_grad()
def center_mean(x: Tensor, dim: Tuple[int, ...]) -> Tensor:
    """Returns the center of the input tensor by calculating the mean.

    Args:
        x:
            Input tensor.
        dim:
            Dimensions along which the mean is calculated.

    Returns:
        The center of the input tensor.
    """
    batch_center = torch.mean(x, dim=dim, keepdim=True)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(batch_center)
        batch_center = batch_center / dist.get_world_size()
    return batch_center


@dataclass
class DINOConfig(SelfDistillationConfig):
    """Configuration for the DINO model parameters.

    Parameters
    ----------
    warmup_teacher_temperature: float
        Initial value of the teacher temperature.
        Should be decreased if the training loss does not decrease.
    teacher_temperature : float
        Final value of the teacher temperature after linear warmup.
    warmup_teacher_temperature_epochs : int
        Number of epochs to warm up the teacher temperature.
    student_temperature : float
        Temperature of the student.
    center_momentum : float
        Momentum used to update the center.
    """

    warmup_teacher_temperature: float = 0.04
    teacher_temperature: float = 0.04
    warmup_teacher_temperature_epochs: int = 30
    student_temperature: float = 0.1
    center_momentum: float = 0.9

    def trainer(self):
        return DINO
