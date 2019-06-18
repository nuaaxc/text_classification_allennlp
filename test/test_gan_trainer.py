from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.params import Params
from allennlp.training.trainer_base import TrainerBase


class GanTrainerTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        params = Params({
            "trainer": {
                "type": "gan-test"
            },
            "data_reader": {"type": "sampling", "sampler": {"type": "normal", "mean": 4.0, "stdev": 1.25}},
            "noise_reader": {"type": "sampling", "sampler": {"type": "uniform"}},
            "generator": {
                "type": "generator-test",
                "input_dim": 1,
                "hidden_dim": 5,
                "output_dim": 1
                },
            "discriminator": {
                "type": "discriminator-test",
                "input_dim": 500,
                "hidden_dim": 10
            },
            "iterator": {
                "type": "basic",
                "batch_size": 500
            },
            "noise_iterator": {
                "type": "basic",
                "batch_size": 500
            },
            "generator_optimizer": {"type": "sgd", "lr": 0.1},
            "discriminator_optimizer": {"type": "sgd", "lr": 0.1},
            "num_epochs": 5,
            "batches_per_epoch": 2
        })

        self.trainer = TrainerBase.from_params(params, self.TEST_DIR)

    def test_gan_can_train(self):
        self.trainer.train()
