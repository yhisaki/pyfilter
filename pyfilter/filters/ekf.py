import torch
import torch.distributions
from torch.autograd.functional import jacobian

from pyfilter.filters.base_filter import BaseFilter
from pyfilter.models.base_model import BaseModel


class EKF(BaseFilter):
    def __init__(
        self,
        model: BaseModel,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
    ) -> None:
        """Extended Kalman Filter

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

    def estimate(self, observed: torch.Tensor, input: torch.Tensor):

        w = torch.zeros(self.model.dim_state)  # process noise
        v = torch.zeros(self.model.dim_observation)  # observation noise

        x_pred = self.model.step(state=self.x_est, input=input, noise=w)
        y_pred = self.model.observe(state=x_pred, input=input, noise=v)

        F, W = jacobian(
            func=lambda x, w: self.model.step(state=x, input=input, noise=w), inputs=(self.x_est, w)
        )
        H, V = jacobian(
            func=lambda x, v: self.model.observe(state=x, input=input, noise=v), inputs=(x_pred, v)
        )

        P_pred = F @ self.P_est @ F.transpose(0, 1) + W @ self.Q @ W.transpose(0, 1)

        K = (
            P_pred
            @ H.transpose(0, 1)
            @ (H @ P_pred @ H.transpose(0, 1) + V @ self.R @ V.transpose(0, 1)).inverse()
        )

        self.x_est = x_pred + K @ (observed - y_pred)
        self.P_est = (torch.eye(self.dim_state) - K @ H) @ self.P_est

        return self.x_est, self.P_est
