from typing import List
import matplotlib.pyplot as plt
import torch


def plot_results(path: str, ts, logs: dict):

    state_names = [
        "theta1",
        "theta2",
        "theta_dot1",
        "theta_dot2",
        "m2",
    ]

    state_names_tex = [
        r"$\theta_1$",
        r"$\theta_2$",
        r"$\dot{\theta}_1$",
        r"$\dot{\theta}_2$",
        r"$m_2$",
    ]

    def _plot(key: str, idx: int, linestyle="-", color="r"):
        if key in logs:
            plt.plot(ts, [v[idx] for v in logs[key]], linestyle, label=key, color=color)

    def _plot_with_scale(key, idx, color):
        if key in logs:
            mean = torch.tensor([v[idx] for v in logs[key]["mean"]])
            std = torch.tensor([torch.sqrt(cov[idx, idx]) for cov in logs[key]["cov"]])
            plt.fill_between(ts, y1=mean - std, y2=mean + std, alpha=0.1, color=color)
            plt.plot(ts, mean, color=color, label=key)
            # print(std)

    for i in range(5):
        _plot_with_scale("ukf", i, color="g")
        _plot_with_scale("ekf", i, color="y")
        _plot("euler_method", i, color="b")
        _plot("true", i, linestyle="--", color="r")

        plt.title(state_names_tex[i])
        plt.legend()
        plt.savefig(f"{path}/{state_names[i]}.png")
        plt.clf()
