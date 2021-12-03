import numpy as np
import torch
import torch.distributions
from plot_results import plot_results

from pyfilter.filters import EKF, UKF
from pyfilter.models import TwoLink1
from pyfilter.simulator import Simulator


def main():
    torch.manual_seed(0)

    T = 10.0
    dT = 0.01

    twolink = TwoLink1(I2=3.0, dt=dT)

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

    ukf = UKF(
        model=twolink,
        x0=initial_state_distribution.loc,
        P0=initial_state_distribution.covariance_matrix,
        Q=state_noise_distribution.covariance_matrix,
        R=observation_noise_distribution.covariance_matrix,
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
    input = torch.tensor([0.0, 0.0])

    ts = np.arange(stop=T, step=dT)

    logs = {
        "true": [],
        "ukf": {"mean": [], "cov": []},
        "ekf": {"mean": [], "cov": []},
    }

    for t in ts:
        x_true = simulator.step(input)
        obs = simulator.observe(input)
        x_est_ukf, P_est_ukf = ukf.estimate(observed=obs, input=input)
        x_est_ekf, P_est_ekf = ekf.estimate(observed=obs, input=input)

        logs["true"].append(x_true)
        logs["ukf"]["mean"].append(x_est_ukf)
        logs["ukf"]["cov"].append(P_est_ukf)
        logs["ekf"]["mean"].append(x_est_ekf)
        logs["ekf"]["cov"].append(P_est_ekf)

    plot_results("results/twolink1/compare_ukf_ekf", ts, logs)


if __name__ == "__main__":
    main()
