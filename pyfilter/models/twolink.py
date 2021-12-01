import torch
import torch.distributions

from pyfilter.models.base_model import BaseModel


class TwoLinkBase(BaseModel):
    dim_state = 4
    dim_input = 2

    def __init__(
        self,
        a1: float = 1.0,
        l1: float = 0.5,
        a2: float = 0.5,
        l2: float = 0.25,
        m1: float = 10.0,
        m2: float = 8.0,
        I1: float = 5.0,
        I2: float = 3.0,
        d1: float = 0.0,
        d2: float = 0.0,
        g: float = 9.8,
        dt: float = 0.01,
    ) -> None:
        self.a1 = a1
        self.l1 = l1
        self.a2 = a2
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.I1 = I1
        self.I2 = I2
        self.d1 = d1
        self.d2 = d2
        self.g = g
        self.dt = dt

    def D(self, x: torch.Tensor):
        phi1 = self.m1 * (self.l1 ** 2) + self.m2 * (self.a1 ** 2) + self.I1
        phi2 = self.m2 * (self.l2 ** 2) + self.I2
        phi3 = self.m2 * self.a1 * self.l2
        c2 = torch.cos(x[1])

        return torch.tensor(
            [
                [phi1 + phi2 + 2 * phi3 * c2, phi2 + phi3 * c2],
                [phi2 + phi3 * c2, phi2],
            ]
        )

    def h(self, x: torch.Tensor):
        phi3 = self.m2 * self.a1 * self.l2
        c1 = torch.cos(x[0])
        s2 = torch.sin(x[1])
        c12 = torch.cos(x[0] + x[1])
        theta_dot = x[2:4]

        theta_dot_coeff = -phi3 * s2 * torch.tensor(
            [[theta_dot[1], theta_dot[0] + theta_dot[1]], [-theta_dot[0], 0.0]]
        ) + torch.tensor([[self.d1, 0], [0, self.d2]])
        v1 = torch.tensor(
            [
                self.m1 * self.l1 * c1 + self.m2 * (self.a1 * c1 + self.l2 * c12),
                self.m2 * self.l2 * c12,
            ]
        )

        return theta_dot_coeff @ theta_dot + self.g * v1

    def dae(self, x: torch.Tensor, u: torch.Tensor, noise: torch.Tensor = torch.zeros(dim_state)):
        """dynamics of twolink model

        Args:
            x (torch.Tensor): state, [theta1, theta2, theta1_dot, theta2_dot]
            u (torch.Tensor): input, [tau_1, tau_2]
            noise (torch.Tensor, optional): state noise. Defaults to torch.zeros(dim_state).

        Returns:
            [type]: [description]
        """
        dxdt = torch.zeros(4)
        dxdt[0:2] = x[2:4]
        dxdt[2:4] = self.D(x).inverse() @ (u - self.h(x))
        return dxdt + noise

    def step(self, state: torch.Tensor, input: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return state + self.dt * self.dae(state, input, noise)


class TwoLink1(TwoLinkBase):
    dim_observation = 1
    dim_state = TwoLinkBase.dim_state
    dim_input = TwoLinkBase.dim_input

    def __init__(
        self,
        a1: float = 1,
        l1: float = 0.5,
        a2: float = 0.5,
        l2: float = 0.25,
        m1: float = 10,
        m2: float = 8,
        I1: float = 5,
        I2: float = 2.0,
        d1: float = 0,
        d2: float = 0,
        g: float = 9.8,
        dt: float = 0.01,
    ) -> None:
        super().__init__(
            a1=a1, l1=l1, a2=a2, l2=l2, m1=m1, m2=m2, I1=I1, I2=I2, d1=d1, d2=d2, g=g, dt=dt
        )

    def observe(
        self,
        state: torch.Tensor,
        input: torch.Tensor = torch.zeros(dim_input),
        noise: torch.Tensor = torch.zeros(dim_observation),
    ):
        return self.a1 * torch.sin(state[0]) + self.a2 * torch.sin(state[0] + state[1]) + noise


class TwoLink2(TwoLink1):
    dim_state = 5

    def __init__(
        self,
        a1: float = 1,
        l1: float = 0.5,
        a2: float = 0.5,
        l2: float = 0.25,
        m1: float = 10,
        I1: float = 5,
        I2: float = 2.0,
        d1: float = 0,
        d2: float = 0,
        g: float = 9.8,
        dt: float = 0.01,
    ) -> None:
        super().__init__(
            a1=a1, l1=l1, a2=a2, l2=l2, m1=m1, m2=None, I1=I1, I2=I2, d1=d1, d2=d2, g=g, dt=dt
        )

    def dae(self, x: torch.Tensor, u: torch.Tensor, noise: torch.Tensor = torch.zeros(dim_state)):
        dxdt = torch.zeros(5)
        self.m2 = x[4]
        dxdt[0:2] = x[2:4]
        dxdt[2:4] = self.D(x).inverse() @ (u - self.h(x))
        return dxdt + noise
