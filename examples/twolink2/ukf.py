import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions

from pyfilter.distributions import Delta
from pyfilter.filters import UKF
from pyfilter.models import TwoLink2
from pyfilter.simulator import Simulator
from plot_results import plot_results


def main():
    torch.manual_seed(1)

    T = 10.0
    dT = 0.01
    m2 = 8.0

    twolink = TwoLink2(I2=3.0, dt=dT)

    initial_state_distribution = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, torch.pi / 4.0, 0.0, 0.0, m2]),
        covariance_matrix=torch.diag(torch.tensor([0.1, 0.1, 0.01, 0.01, 1.0])),
    )

    state_noise_distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(5),
        covariance_matrix=torch.diag(torch.tensor([0.001, 0.001, 0.01, 0.01, 1e-20])),
    )

    observation_noise_distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(1),
        covariance_matrix=torch.diag(torch.tensor([0.01 ** 2])),
    )

    simulator = Simulator(
        model=twolink,
        initial_state_distribution=initial_state_distribution,
        state_noise_distribution=state_noise_distribution,
        observation_noise_distribution=observation_noise_distribution,
    )

    ukf = UKF(
        model=twolink,
        x0=initial_state_distribution.loc,
        P0=initial_state_distribution.covariance_matrix,
        Q=state_noise_distribution.covariance_matrix,
        R=observation_noise_distribution.covariance_matrix,
    )

    euler_method = Simulator(
        model=twolink,
        initial_state_distribution=Delta(initial_state_distribution.loc),
        state_noise_distribution=Delta(state_noise_distribution.loc),
        observation_noise_distribution=(Delta(observation_noise_distribution.loc)),
    )

    input = torch.tensor([0.0, 0.0])

    ts = np.arange(stop=T, step=dT)

    logs = {"true": [], "ukf": {"mean": [], "cov": []}, "euler_method": []}

    for t in ts:
        x_true = simulator.step(input)
        obs = simulator.observe(input)
        x_est, P_est = ukf.estimate(observed=obs, input=input)
        x_eul = euler_method.step(input)

        logs["true"].append(x_true)
        logs["ukf"]["mean"].append(x_est)
        logs["ukf"]["cov"].append(P_est)
        logs["euler_method"].append(x_eul)

    plot_results("results/twolink2/ukf", ts, logs)


if __name__ == "__main__":
    main()
