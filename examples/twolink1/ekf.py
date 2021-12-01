import numpy as np
import torch
import torch.distributions
from plot_results import plot_results

from pyfilter.distributions import Delta
from pyfilter.filters import EKF
from pyfilter.models import TwoLink1
from pyfilter.simulator import Simulator


def main():
    torch.manual_seed(0)

    T = 10.0
    dT = 0.01

    twolink = TwoLink1(dt=dT)

    initial_state_distribution = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, torch.pi / 4.0, 0.0, 0.0]),
        covariance_matrix=torch.diag(torch.tensor([0.5, 0.5, 0.1, 0.1])),
    )

    state_noise_distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(4),
        covariance_matrix=torch.diag(torch.tensor([0.001, 0.001, 0.01, 0.01])),
    )

    observation_noise_distribution = torch.distributions.MultivariateNormal(
        loc=torch.zeros(1),
        covariance_matrix=torch.diag(torch.tensor([0.01 ** 2])),
    )

    ekf = EKF(
        model=twolink,
        x0=initial_state_distribution.loc,
        P0=initial_state_distribution.covariance_matrix,
        Q=state_noise_distribution.covariance_matrix,
        R=observation_noise_distribution.covariance_matrix,
    )

    simulator = Simulator(
        model=twolink,
        initial_state_distribution=initial_state_distribution,
        state_noise_distribution=state_noise_distribution,
        observation_noise_distribution=observation_noise_distribution,
    )

    euler_method = Simulator(
        model=twolink,
        initial_state_distribution=Delta(initial_state_distribution.loc),
        state_noise_distribution=Delta(state_noise_distribution.loc),
        observation_noise_distribution=(Delta(observation_noise_distribution.loc)),
    )

    input = torch.tensor([0.0, 0.0])

    ts = np.arange(stop=T, step=dT)

    logs = {"true": [], "ekf": {"mean": [], "cov": []}, "euler_method": []}

    for t in ts:
        x_true = simulator.step(input)
        obs = simulator.observe(input)
        x_est, P_est = ekf.estimate(observed=obs, input=input)
        x_eul = euler_method.step(input)

        logs["true"].append(x_true)
        logs["ekf"]["mean"].append(x_est)
        logs["ekf"]["cov"].append(P_est)
        logs["euler_method"].append(x_eul)

    plot_results("results/twolink1/ekf", ts, logs)


if __name__ == "__main__":
    main()
