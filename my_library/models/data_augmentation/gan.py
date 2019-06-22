from allennlp.models import Model
from allennlp.common.params import Params


@Model.register("gan")
class Gan(Model):
    """
    Our trainer wants a single model, so we cheat by encapsulating both the
    generator and discriminator inside a single model. We'll access them individually.
    """

    # pylint: disable=abstract-method
    def __init__(self,
                 feature_extractor: Model,
                 generator: Model,
                 discriminator: Model,
                 classifier: Model) -> None:
        super().__init__(None)

        # We need our optimizer to know which parameters came from
        # which model, so we cheat by adding tags to them.
        for param in feature_extractor.parameters():
            setattr(param, '_feature_extractor', True)
        for param in generator.parameters():
            setattr(param, '_generator', True)
        for param in discriminator.parameters():
            setattr(param, '_discriminator', True)
        for param in classifier.parameters():
            setattr(param, '_classifier', True)

        self.feature_extractor = feature_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
