import math
from typing import List

import torch
import torch.distributions

from pyfilter.filters.base_filter import BaseFilter
from pyfilter.models.base_model import BaseModel


class UKF(BaseFilter):
    def __init__(
        self,
        model: BaseModel,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
    ) -> None:
        """Unscented Kalman Filter

        Args:
            model (BaseModel): model
            x0 (torch.Tensor): mean of estimated initial state
            P0 (torch.Tensor): covariance matrix of estimated initial state
            Q (torch.Tensor): covariance matrix of process noise
            R (torch.Tensor): covariance matrix of observation noise
        """
        self.model = model
        self.x_est = x0
        self.P_est = P0
        self.Q = Q
        self.R = R

        self.dim_state = len(x0)
        self.dim_obs = len(R[0, :])

    def get_sigma_points(self, x, P: torch.Tensor):
        n = len(x)
        kappa = 3 - n
        Chi: List[torch.Tensor] = []
        W: List[float] = []
        Chi.append(x)
        W.append(kappa / (n + kappa))
        rootP: torch.Tensor = torch.linalg.cholesky(P)
        for i in range(len(x)):
            chi_i_p = x + math.sqrt(n + kappa) * rootP[:, i]
            chi_i_m = x - math.sqrt(n + kappa) * rootP[:, i]
            w = 1.0 / (2.0 * (n + kappa))
            Chi.extend([chi_i_p, chi_i_m])
            W.extend([w, w])

        return Chi, W

    def estimate(self, observed: torch.Tensor, input: torch.Tensor):
        xa = torch.zeros(2 * self.dim_state)  # augumented state
        Pa = torch.zeros((2 * self.dim_state, 2 * self.dim_state))

        xa[0 : self.dim_state] = self.x_est
        Pa[0 : self.dim_state, 0 : self.dim_state] = self.P_est
        Pa[self.dim_state : 2 * self.dim_state, self.dim_state : 2 * self.dim_state] = self.Q

        Chi, W = self.get_sigma_points(xa, Pa)

        Chi_pred: List[torch.Tensor] = []
        x_pred = torch.zeros_like(self.x_est)

        for chi, w in zip(Chi, W):
            chi_pred = self.model.step(
                chi[0 : self.dim_state], input, chi[self.dim_state : 2 * self.dim_state]
            )
            Chi_pred.append(chi_pred)
            x_pred += w * chi_pred

        Eta: List[torch.Tensor] = []
        y_pred = torch.zeros_like(observed)
        P_pred = torch.zeros_like(self.P_est)
        for chi_pred, w in zip(Chi_pred, W):
            P_pred += w * (chi_pred - x_pred)[..., None] @ (chi_pred - x_pred)[None, ...]
            eta = self.model.observe(chi_pred, input)
            Eta.append(eta)
            y_pred += w * eta

        Pyy_pred = torch.zeros_like(self.R)
        for eta, w in zip(Eta, W):
            Pyy_pred += w * (eta - y_pred)[..., None] @ (eta - y_pred)[None, ...]
        Pvv = self.R + Pyy_pred
        Pxv = torch.zeros((self.dim_state, self.dim_obs))

        for chi_pred, eta, w in zip(Chi_pred, Eta, W):
            Pxv += w * (chi_pred - x_pred)[..., None] @ (eta - y_pred)[None, ...]

        G = Pxv @ Pvv.inverse()
        self.x_est = x_pred + G @ (observed - y_pred)
        self.P_est = P_pred - G @ Pvv @ G.transpose(0, 1)
        return self.x_est, self.P_est
