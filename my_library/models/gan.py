from typing import Dict
from allennlp.models import Model
from allennlp.common.params import Params


@Model.register("gan-base")
class Gan(Model):
    """
    Our trainer wants a single model, so we cheat by encapsulating both the
    generator and discriminator inside a single model. We'll access them individually.
    """

    # pylint: disable=abstract-method
    def __init__(self,
                 generator: Model,
                 discriminator: Model) -> None:
        super().__init__(None)

        # We need our optimizer to know which parameters came from
        # which model, so we cheat by adding tags to them.
        for param in generator.parameters():
            setattr(param, '_generator', True)
        for param in discriminator.parameters():
            setattr(param, '_discriminator', True)

        self.generator = generator
        self.discriminator = discriminator

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_generator = self.generator.get_metrics(reset=reset)
        metrics_discriminator = self.discriminator.get_metrics(reset=reset)
        metrics = {}
        metrics.update(metrics_generator)
        metrics.update(metrics_discriminator)
        return metrics
