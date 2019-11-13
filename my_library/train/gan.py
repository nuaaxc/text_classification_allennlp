from typing import Dict, Iterable, Any, Tuple, Union
import logging
import time
import os
import datetime
import numpy as np
import random
import scipy.stats

import sklearn
import torch
import torch.nn.functional as F

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.common.util import dump_metrics
from allennlp.data import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models import Model
from allennlp.nn import util as nn_util
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.checkpointer import Checkpointer
from allennlp.training import util as training_util

from my_library.optimisation import GanOptimizer
from my_library.models.data_augmentation import Gan
from config import DirConfig

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def aug_normal(f, cuda_device):
    return f + 0.1 * torch.randn_like(f).cuda(cuda_device)


# def aug_uniform(f, cuda_device):
#     return f + 0.001 * torch.rand_like(f).cuda(cuda_device)


def compute_gradient_penalty(D, h_real, h_fake, label, cuda_device):
    """Calculates the gradient penalty loss for WGAN GP"""

    alpha = torch.rand(h_real.size(0), 1).cuda(cuda_device)
    differences = h_fake - h_real
    interpolates = h_real + (alpha * differences)
    interpolates = interpolates.cuda(cuda_device).requires_grad_()
    d_interpolates = D(interpolates, label)["output"]

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty


@TrainerBase.register("gan")
class GanTrainer(TrainerBase):
    def __init__(self,
                 model: Gan,
                 optimizer: torch.optim.Optimizer,

                 noise_dataset: Iterable[Instance] = None,
                 feature_dataset: Iterable[Instance] = None,

                 noise_iterator: DataIterator = None,
                 feature_iterator: DataIterator = None,
                 validation_metric: str = "-loss",
                 serialization_dir: str = None,
                 n_epoch_gan: int = 0,
                 batch_size: int = 16,
                 cuda_device: int = 0,
                 patience: int = 5,
                 conservative_rate: float = 1.0,
                 num_serialized_models_to_keep: int = 1,
                 keep_serialized_model_every_num_seconds: int = None,
                 num_loop_discriminator: int = 5,
                 batch_per_epoch: int = 10,
                 batch_per_generator: int = 10,
                 gen_step: int = 10,
                 clip_value: float = 1,
                 ) -> None:
        super().__init__(serialization_dir, cuda_device)

        self.model = model

        self.optimizer = optimizer

        self.noise_dataset = noise_dataset
        self.feature_dataset = feature_dataset

        self.noise_iterator = noise_iterator
        self.feature_iterator = feature_iterator

        self.n_epoch_gan = n_epoch_gan

        self._batch_size = batch_size

        self._gan_loss_tracker = MetricTracker(patience, validation_metric)

        self._checkpointer = Checkpointer(serialization_dir,
                                          keep_serialized_model_every_num_seconds,
                                          num_serialized_models_to_keep)

        self.conservative_rate = conservative_rate
        self.num_loop_discriminator = num_loop_discriminator
        self.batch_per_epoch = batch_per_epoch
        self.batch_per_generator = batch_per_generator
        self.gen_step = gen_step

        self.cuda_device = cuda_device
        self.clip_value = clip_value
        self.n_epochs = None

        self.aug_features = []

    def _train_discriminator(self, f, noise, label):
        self.optimizer.stage = 'discriminator'
        self.optimizer.zero_grad()

        for p in self.model.discriminator.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        # Real example
        real_validity = self.model.discriminator(f, label)["output"]

        # Fake example
        fake_data = self.model.generator(f, noise, label=label)["output"].detach()
        fake_validity = self.model.discriminator(fake_data, label)["output"]

        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(self.model.discriminator,
                                                    h_real=f.data,
                                                    h_fake=fake_data.data,
                                                    label=label,
                                                    cuda_device=self.cuda_device)

        d_error = -torch.mean(real_validity) + torch.mean(fake_validity) + 10. * gradient_penalty

        d_error.backward()
        self.optimizer.step()

        # Clip weights of discriminator
        # for p in self.model.discriminator.parameters():
        #     p.data.clamp_(-self.clip_value, self.clip_value)

        return d_error.item()

    def _train_generator(self, f, noise, label):
        self.optimizer.stage = 'generator'
        self.optimizer.zero_grad()

        for p in self.model.discriminator.parameters():
            p.requires_grad = False

        generated = self.model.generator(feature=f,
                                         noise=noise,
                                         label=label,
                                         discriminator=self.model.discriminator)

        cls_results = self.model.classifier(generated['output'], label)
        cls_prediction_syn = cls_results['output']
        cls_prediction_real = self.model.classifier(f, label)['output']
        kl = F.kl_div(cls_prediction_syn.log(), cls_prediction_real, reduction='batchmean')

        fake_error = generated["loss"] + 10 * cls_results['loss'] + 10 * kl
        fake_error.backward()

        self.optimizer.step()

        return fake_error.item()

    def sample_feature(self, feature_iterator, conservative_rate):
        """
        Sample from a real feature or a generated feature
        """
        choice: float = random.random()
        if len(self.aug_features) == 0 or choice > conservative_rate:
            feature = next(feature_iterator)
        else:
            feature = random.sample(self.aug_features, 1)[0]
        feature = nn_util.move_to_device(feature, self.cuda_device)
        return feature

    def _train_gan(self) -> Any:
        loss_d = []
        loss_g = []
        feature_iterator = self.feature_iterator(self.feature_dataset, num_epochs=None, shuffle=True)
        noise_iterator = self.noise_iterator(self.noise_dataset)
        # Freeze the classifier
        for p in self.model.classifier.parameters():
            p.requires_grad = False

        # Freeze the feature extractor
        for p in self.model.feature_extractor.parameters():
            p.requires_grad = False

        # train on this epoch
        for i in range(self.batch_per_epoch):
            f = self.sample_feature(feature_iterator, self.conservative_rate)
            noise = next(noise_iterator)['array']
            noise = nn_util.move_to_device(noise, self.cuda_device)
            # ##############
            # discriminator
            # ##############
            _loss_d = self._train_discriminator(f['feature'], noise, f['label'])
            loss_d.append(_loss_d)

            if (i + 1) % self.num_loop_discriminator == 0:
                f = self.sample_feature(feature_iterator, self.conservative_rate)
                noise = next(noise_iterator)['array']
                noise = nn_util.move_to_device(noise, self.cuda_device)
                # ##########
                # generator
                # ##########
                _loss_g = self._train_generator(f['feature'], noise, f['label'])
                loss_g.append(_loss_g)

        # #################
        # generate samples
        # #################
        g_data = []
        g_label = []
        print(len(self.aug_features))

        for i in range(self.batch_per_generator):
            f = self.sample_feature(feature_iterator, self.conservative_rate)
            noise = next(noise_iterator)['array']
            noise = nn_util.move_to_device(noise, self.cuda_device)
            generated = self.model.generator(f['feature'], noise, f['label'])['output']
            self.aug_features.append({'feature': generated.data.cpu(),
                                      'label': f['label'].data.cpu()})

            g_data.append(generated.data.cpu().numpy())
            g_label.extend(f['label'].data.cpu().numpy())
        return np.mean(loss_d), np.mean(loss_g), np.vstack(g_data), g_label

    def feature_generation(self, K):
        aug_features = []
        feature_iterator = self.feature_iterator(self.feature_dataset, num_epochs=None, shuffle=True)
        noise_iterator = self.noise_iterator(self.noise_dataset)
        for k in range(K):
            for n in range(self.batch_per_epoch):
                if k == 0:
                    f = next(feature_iterator)
                    aug_features.append({'feature': f['feature'], 'label': f['label']})
                else:
                    noise = next(noise_iterator)['array']
                    noise = nn_util.move_to_device(noise, self.cuda_device)

                    f = random.sample(aug_features, 1)[0]
                    f = nn_util.move_to_device(f, self.cuda_device)

                    generated = self.model.generator(f['feature'], noise, f['label'])['output']
                    aug_features.append({'feature': generated.data.cpu(),
                                         'label': f['label'].data.cpu()})
        return aug_features

    def _train_epoch(self, epoch: int) -> Any:

        logger.info("Epoch %d/%d", epoch, self.n_epochs)
        logger.info("Phase: %s ..." % self.phase)
        self.model.train()

        metrics = {}

        loss_d, loss_g, g_data, g_label = self._train_gan()
        metrics.update({'d_loss': loss_d})
        metrics.update({'g_loss': loss_g})
        metrics.update({'gan_loss': 0.5 * (loss_d + loss_g)})
        metrics.update({'g_data': g_data})
        metrics.update({'g_label': g_label})
        logger.info('[d_loss] %s, [g_loss] %s, [mean] %s' % (loss_d, loss_g, 0.5 * (loss_d + loss_g)))

        return metrics

    def train(self) -> Any:
        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        self.n_epochs = self.n_epoch_gan
        logger.info('[Load real model] ...')
        best_model_state = torch.load(os.path.join(self.model_real_dir, 'best.th'))
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            logger.info('[Loaded]')

        g_data_epochs = {}  # training data (generated)
        d_loss_epochs = {}
        g_loss_epochs = {}
        gan_loss_epochs = {}

        training_features = []

        # train over epochs
        for epoch in range(self.n_epochs):

            epoch_start_time = time.time()

            # train (over one epoch)
            train_metrics = self._train_epoch(epoch)

            d_loss_epochs[epoch] = train_metrics['d_loss']
            g_loss_epochs[epoch] = train_metrics['g_loss']
            gan_loss_epochs[epoch] = train_metrics['gan_loss']
            g_data_epochs[epoch] = (train_metrics['g_data'], train_metrics['g_label'])

            if self.phase == 'gan':
                self._gan_loss_tracker.add_metric(gan_loss_epochs[epoch])

            #########################
            # Create overall metrics
            #########################
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                if key != 'r_data' and key != 'g_data' and key != 'r_label' and key != 'g_label':
                    metrics["training_" + key] = value

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            #############
            # Save model
            #############
            self._save_checkpoint(epoch, self._gan_loss_tracker)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self.n_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * (self.n_epochs / float(epoch + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        meta_data = {
            'metrics': metrics,
            'g_data_epochs': g_data_epochs,
            'd_loss_epochs': d_loss_epochs,
            'g_loss_epochs': g_loss_epochs,
            'training_features': training_features
        }
        return meta_data

    def _save_checkpoint(self, epoch: Union[int, str], _tracker) -> None:
        self._checkpointer.save_checkpoint(
            model_state=self.model.state_dict(),
            epoch=epoch,
            training_states={},
            is_best_so_far=_tracker.is_best_so_far())

    @classmethod
    def from_params(cls,  # type: ignore
                    params: Params,
                    serialization_dir: str,
                    recover: bool = False,
                    cache_directory: str = None,
                    cache_prefix: str = None
                    ) -> 'GanTrainer':

        train_feature = params.pop('train_feature_path')

        config_file = params.pop('config_file')
        cuda_device = params.pop_int("cuda_device")

        # Data reader

        noise_reader = DatasetReader.from_params(params.pop("noise_reader"))
        noise_dataset = noise_reader.read("")

        feature_reader = DatasetReader.from_params(params.pop("feature_reader"))
        feature_dataset = feature_reader.read("")

        # Vocabulary
        # vocab = Vocabulary.from_files(os.path.join(serialization_dir, "vocabulary"))
        vocab = Vocabulary.from_instances(feature_dataset)

        # Iterator
        noise_iterator = DataIterator.from_params(params.pop("noise_iterator"))
        noise_iterator.index_with(vocab)
        feature_iterator = DataIterator.from_params(params.pop("feature_iterator"))
        noise_iterator.index_with(vocab)

        # Model
        generator = Model.from_params(params.pop("generator"), vocab=None).cuda(cuda_device)
        discriminator = Model.from_params(params.pop("discriminator"), vocab=None).cuda(cuda_device)

        cls_model = Model.from_params(params.pop("cls_model")).cuda(cuda_device)
        cls_model.load_state_dict(torch.load(params.pop("best_cls_model_state_path")))

        model = Gan(
            generator=generator,
            discriminator=discriminator,
        )

        # Optimize
        parameters = []
        for component in [
            model.generator,
            model.discriminator]:
            parameters += [[n, p] for n, p in component.named_parameters() if p.requires_grad]
        optimizer = GanOptimizer.from_params(parameters, params.pop("optimizer"))

        n_epoch_gan = params.pop_int("n_epoch_gan")
        batch_size = params.pop_int("batch_size")
        patience = params.pop_int("patience")
        conservative_rate = params.pop_float("conservative_rate")
        num_loop_discriminator = params.pop_int("num_loop_discriminator")
        batch_per_epoch = params.pop_int("batch_per_epoch")
        batch_per_generator = params.pop_int("batch_per_generator")
        gen_step = params.pop_int("gen_step")
        clip_value = params.pop_int("clip_value")
        params.pop("trainer")
        params.assert_empty(__name__)

        return cls(model=model,
                   optimizer=optimizer,

                   noise_dataset=noise_dataset,
                   feature_dataset=feature_dataset,

                   noise_iterator=noise_iterator,
                   feature_iterator=feature_iterator,

                   serialization_dir=serialization_dir,

                   n_epoch_gan=n_epoch_gan,

                   batch_size=batch_size,
                   cuda_device=cuda_device,
                   patience=patience,
                   conservative_rate=conservative_rate,
                   num_loop_discriminator=num_loop_discriminator,
                   batch_per_epoch=batch_per_epoch,
                   batch_per_generator=batch_per_generator,
                   gen_step=gen_step,
                   clip_value=clip_value)
