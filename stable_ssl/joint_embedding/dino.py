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

    def compute_ssl_loss(self, z_i, z_j):
        """Compute the loss of the DINO model.

        Parameters
        ----------
        z_i : torch.Tensor
            Latent representation of the first augmented view of the batch.
        z_j : torch.Tensor
            Latent representation of the second augmented view of the batch.

        Returns
        -------
        float
            The computed loss.
        """

        # def compute_ssl_loss(self, projections):
        #     # Construct target with the target ('teacher') network.
        #     with torch.no_grad():
        #         global_views = self.data[0][:2]  # First two views are global views.
        #         projections_target = [
        #             self.projector_target(self.backbone_target(view))
        #             for view in global_views
        #         ]

        #     if epoch < self.warmup_teacher_temp_epochs:
        #         teacher_temp = self.teacher_temp_schedule[epoch]
        #     else:
        #         teacher_temp = self.teacher_temp

        #     teacher_out = torch.stack(teacher_out)
        #     t_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1)

        #     student_out = torch.stack(student_out)
        #     s_out = F.log_softmax(student_out / self.student_temp, dim=-1)

        #     # Calculate feature similarities, ignoring the diagonal
        #     # b = batch_size, t = n_views_teacher, s = n_views_student, d = output_dim
        #     loss = -torch.einsum("tbd,sbd->ts", t_out, s_out)
        #     loss.fill_diagonal_(0)

        #     # Number of loss terms, ignoring the diagonal
        #     n_terms = loss.numel() - loss.diagonal().numel()
        #     batch_size = teacher_out.shape[1]

        #     loss = loss.sum() / (n_terms * batch_size)

        #     # Update the center used for the teacher output
        #     self.update_center(teacher_out)

        #     return loss
