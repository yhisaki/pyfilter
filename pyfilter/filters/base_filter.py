from abc import ABCMeta, abstractmethod

import torch


class BaseFilter(object, metaclass=ABCMeta):
    @abstractmethod
    def estimate(self, observed: torch.Tensor, input: torch.Tensor = ...) -> torch.Tensor:
        raise NotImplementedError()
