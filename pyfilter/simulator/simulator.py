import torch
import torch.distributions

from pyfilter.models.base_model import BaseModel


class Simulator(object):
    def __init__(
        self,
        model: BaseModel,
        initial_state_distribution: torch.distributions.Distribution,
        state_noise_distribution: torch.distributions.Distribution,
        observation_noise_distribution: torch.distributions.Distribution,
    ) -> None:
        super().__init__()
        self.model = model
        self.state = initial_state_distribution.sample()
        self.state_noise_distribution = state_noise_distribution
        self.observation_noise_distribution = observation_noise_distribution

    def step(self, input: torch.Tensor):
        self.state = self.model.step(self.state, input, self.state_noise_distribution.sample())
        return self.state

    def observe(self, input=...):
        return self.model.observe(
            self.state, input, noise=self.observation_noise_distribution.sample()
        )
