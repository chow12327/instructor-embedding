from abc import ABC, abstractmethod
import random

import numpy as np
import torch

class Evaluator(ABC):
    """
    Base class for all evaluators
    Extend this class and implement __call__ for custom evaluators.
    """
    def __init__(self, seed=42, **kwargs):
        self.seed = torch.initial_seed()
        print("Seed set at evaluate_model level")
        #commented by Nidhi
        # self.seed = seed
        # random.seed(self.seed)
        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)

    @abstractmethod
    def __call__(self, model):
        """
        This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        pass
