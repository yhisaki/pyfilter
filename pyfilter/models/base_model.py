from abc import ABCMeta, abstractmethod, abstractproperty

import torch


class BaseModel(object, metaclass=ABCMeta):
    @abstractmethod
    def step(
        self, state: torch.Tensor, input: torch.Tensor = ..., noise: torch.Tensor = ...
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def observe(
        self, state: torch.Tensor, input: torch.Tensor = ..., noise: torch.Tensor = ...
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractproperty
    def dim_state(self) -> int:
        pass

    @abstractproperty
    def dim_observation(self) -> int:
        pass

    @abstractproperty
    def dim_input(self) -> int:
        pass
