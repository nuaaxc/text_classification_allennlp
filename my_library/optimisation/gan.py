from typing import List

import torch

from allennlp.common.params import Params
from allennlp.training.optimizers import Optimizer


@Optimizer.register("gan")
class GanOptimizer(torch.optim.Optimizer):
    """
    Similarly, we want different optimizers for the generator and discriminator,
    so we cheat by encapsulating both in a single optimizer that has a state
    indicating which one to use.
    """
    # pylint: disable=super-init-not-called,arguments-differ
    def __init__(self,
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 classifier_optimizer: torch.optim.Optimizer) -> None:
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.classifier_optimizer = classifier_optimizer
        self.stage = ""

    def step(self, _closure=None) -> None:
        if "discriminator" in self.stage:
            self.discriminator_optimizer.step(_closure)
        elif "generator" in self.stage:
            self.generator_optimizer.step(_closure)
        elif "classifier" in self.stage:
            self.classifier_optimizer.step(_closure)
        else:
            raise ValueError("unknown stage")

    def zero_grad(self) -> None:
        if "discriminator" in self.stage:
            self.discriminator_optimizer.zero_grad()
        elif "generator" in self.stage:
            self.generator_optimizer.zero_grad()
        elif "classifier" in self.stage:
            self.classifier_optimizer.zero_grad()
        else:
            raise ValueError("unknown stage")

    @classmethod
    def from_params(cls, parameters: List, params: Params) -> 'GanOptimizer':
        # Because we "tagged" the parameters, we can use getattr to figure out
        # which ones go with which model.

        # print(params.as_dict())
        # print(params.pop("generator_optimizer").as_dict())
        # exit()
        generator_parameters = [("", param) for name, param in parameters
                                if hasattr(param, '_generator')]
        discriminator_parameters = [("", param) for name, param in parameters
                                    if hasattr(param, '_discriminator') or hasattr(param, '_feature_extractor')]
        classifier_parameters = [("", param) for name, param in parameters
                                 if hasattr(param, '_feature_extractor') or hasattr(param, '_classifier')]

        generator_optimizer = Optimizer.from_params(generator_parameters,
                                                    params.pop("generator_optimizer"))
        discriminator_optimizer = Optimizer.from_params(discriminator_parameters,
                                                        params.pop("discriminator_optimizer"))
        classifier_optimizer = Optimizer.from_params(classifier_parameters,
                                                     params.pop("classifier_optimizer"))

        return cls(generator_optimizer=generator_optimizer,
                   discriminator_optimizer=discriminator_optimizer,
                   classifier_optimizer=classifier_optimizer)
