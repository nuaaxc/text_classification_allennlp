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
from my_library.models.gan import Gan

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


@TrainerBase.register("gan-bert")
class GanBertTrainer(TrainerBase):
    def __init__(self,
                 model: Gan,
                 optimizer: torch.optim.Optimizer,

                 train_dataset: Iterable[Instance],
                 noise_dataset: Iterable[Instance] = None,
                 feature_dataset: Iterable[Instance] = None,
                 validation_dataset: Iterable[Instance] = None,
                 test_dataset: Iterable[Instance] = None,

                 data_iterator: DataIterator = None,
                 noise_iterator: DataIterator = None,
                 feature_iterator: DataIterator = None,

                 validation_metric: str = "-loss",
                 serialization_dir: str = None,
                 n_epoch_real: int = 0,
                 n_epoch_gan: int = 0,
                 n_epoch_fake: int = 0,
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
                 n_classes: int = 0,
                 phase: str = None,
                 model_real_dir: str = None,
                 model_gan_dir: str = None,
                 model_fake_dir: str = None,
                 ) -> None:
        super().__init__(serialization_dir, cuda_device)

        self.model = model

        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.noise_dataset = noise_dataset
        self.feature_dataset = feature_dataset

        self.data_iterator = data_iterator
        self.noise_iterator = noise_iterator
        self.feature_iterator = feature_iterator

        self.n_epoch_real = n_epoch_real
        self.n_epoch_gan = n_epoch_gan
        self.n_epoch_fake = n_epoch_fake

        self._batch_size = batch_size
        self.n_classes = n_classes

        self._val_loss_tracker = MetricTracker(patience, validation_metric)
        self._gan_loss_tracker = MetricTracker(patience, validation_metric)
        self._validation_metric = validation_metric[1:]

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
        self.phase = phase
        self.n_epochs = None
        self.model_real_dir = model_real_dir
        self.model_gan_dir = model_gan_dir
        self.model_fake_dir = model_fake_dir

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

    def _train_epoch_on_real(self):
        logger.info('### Training on real data ###')

        cls_loss = 0.
        batches_this_loop = 0

        self.optimizer.stage = 'classifier'

        data_iterator = self.data_iterator(self.train_dataset)
        loop = Tqdm.tqdm(range(self.batch_per_epoch))

        r_data = []
        r_label = []

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            batch = next(data_iterator)
            batch = nn_util.move_to_device(batch, self.cuda_device)

            features = self.model.feature_extractor(batch['tokens'])

            r_data.append(features.data.cpu().numpy())
            r_label.extend(batch['label'].data.cpu().numpy())

            cls_error = self.model.classifier(features, batch['label'])['loss']
            cls_error.backward()

            cls_loss += cls_error.mean().item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'cls_loss_on_real': cls_loss / batches_this_loop}),
                refresh=False
            )
        return {'cls_loss_on_real': cls_loss / batches_this_loop}, np.vstack(r_data), r_label

    def _train_epoch_on_fake(self):
        logger.info('### Training on fake data ###')

        cls_loss = 0.
        batches_this_loop = 0

        self.optimizer.stage = 'classifier'

        loop = Tqdm.tqdm(range(len(self.aug_features)))

        g_data = []
        g_label = []

        for _ in loop:
            batches_this_loop += 1

            self.optimizer.zero_grad()

            f = random.sample(self.aug_features, 1)[0]
            f = nn_util.move_to_device(f, self.cuda_device)
            cls_error = self.model.classifier(f['feature'], f['label'])['loss']

            g_data.append(f['feature'].data.cpu().numpy())
            g_label.extend(f['label'].data.cpu().numpy())

            cls_error.backward()
            cls_loss += cls_error.mean().item()

            self.optimizer.step()

            # Update the description with the latest metrics
            loop.set_description(
                training_util.description_from_metrics({'cls_loss_on_fake': cls_loss / batches_this_loop}),
                refresh=False
            )

        return {'cls_loss_on_fake': cls_loss / batches_this_loop}, np.vstack(g_data), g_label

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

        if self.phase == 'gan':  # train the GAN
            loss_d, loss_g, g_data, g_label = self._train_gan()
            metrics.update({'d_loss': loss_d})
            metrics.update({'g_loss': loss_g})
            metrics.update({'gan_loss': 0.5 * (loss_d + loss_g)})
            metrics.update({'g_data': g_data})
            metrics.update({'g_label': g_label})
            logger.info('[d_loss] %s, [g_loss] %s, [mean] %s' % (loss_d, loss_g, 0.5 * (loss_d + loss_g)))

        elif self.phase == 'real':  # train the classifier on real data
            loss_cls_on_real, r_data, r_label = self._train_epoch_on_real()
            metrics.update(self.model.get_metrics(reset=True))
            metrics.update(loss_cls_on_real)
            metrics.update({'r_data': r_data})
            metrics.update({'r_label': r_label})

        elif self.phase == 'fake':  # train the classifier on fake data
            loss_cls_on_fake, g_data, g_label = self._train_epoch_on_fake()
            metrics.update(loss_cls_on_fake)
            metrics.update({'g_data': g_data})
            metrics.update({'g_label': g_label})

        else:
            raise ValueError('unknown training phase name %s.' % self.phase)

        return metrics

    def test(self) -> Any:
        logger.info("### Testing ###")
        logger.info('[Load best model] ...')
        if self._serialization_dir:
            best_model_state = self._checkpointer.best_model_state()
            if best_model_state:
                self.model.load_state_dict(best_model_state)
                logger.info('[Loaded]')

        if self.model:

            with torch.no_grad():
                self.model.eval()
                y_true = []
                y_pred = []
                r_data = []

                for batch in self.data_iterator(self.test_dataset, num_epochs=1, shuffle=False):
                    batch = nn_util.move_to_device(batch, self.cuda_device)
                    features = self.model.feature_extractor(batch['tokens'])

                    r_data.append(features.data.cpu().numpy())

                    y_ = self.model.classifier(features)['output']

                    y_true.extend(batch['label'].data.cpu().numpy())
                    y_pred.extend(y_.max(1)[1].cpu().numpy())

                return {'r_data': (np.vstack(r_data), y_true),
                        'micro': sklearn.metrics.f1_score(y_true, y_pred, average='micro'),
                        'macro': sklearn.metrics.f1_score(y_true, y_pred, average='macro'),
                        'accuracy': sklearn.metrics.accuracy_score(y_true, y_pred)}

    def train(self) -> Any:
        logger.info("Beginning training.")

        val_metrics: Dict[str, float] = {}
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        if self.phase == 'gan':
            self.n_epochs = self.n_epoch_gan
            logger.info('[Load real model] ...')
            best_model_state = torch.load(os.path.join(self.model_real_dir, 'best.th'))
            if best_model_state:
                self.model.load_state_dict(best_model_state)
                logger.info('[Loaded]')

        if self.phase == 'real':
            self.n_epochs = self.n_epoch_real

        if self.phase == 'fake':
            self.n_epochs = self.n_epoch_fake
            logger.info('[Load gan model] ...')
            best_model_state = torch.load(os.path.join(self.model_gan_dir, 'best.th'))
            if best_model_state:
                self.model.load_state_dict(best_model_state)
                logger.info('[Loaded]')
                # freeze generator
                for p in self.model.generator.parameters():
                    p.requires_grad = False
                # Unfreeze the classifier
                for p in self.model.classifier.parameters():
                    p.requires_grad = True
                # freeze the feature extractor
                for p in self.model.feature_extractor.parameters():
                    p.requires_grad = False
                logger.info('[Feature generation] generating features ...')
                self.aug_features = self.feature_generation(self.gen_step)
                logger.info('[Feature generation] %s features generated.' % len(self.aug_features))

        r_data_epochs = {}  # training data (real)
        g_data_epochs = {}  # training data (generated)
        v_data_epochs = None  # validation data
        d_loss_epochs = {}
        g_loss_epochs = {}
        gan_loss_epochs = {}
        cls_loss_on_real_epochs = {}
        training_features = []

        # train over epochs
        for epoch in range(self.n_epochs):

            epoch_start_time = time.time()

            # train (over one epoch)
            train_metrics = self._train_epoch(epoch)

            if self.phase == 'real':
                r_data_epochs[epoch] = (train_metrics['r_data'], train_metrics['r_label'])
                cls_loss_on_real_epochs[epoch] = train_metrics['cls_loss_on_real']

            elif self.phase == 'gan':
                d_loss_epochs[epoch] = train_metrics['d_loss']
                g_loss_epochs[epoch] = train_metrics['g_loss']
                gan_loss_epochs[epoch] = train_metrics['gan_loss']
                g_data_epochs[epoch] = (train_metrics['g_data'], train_metrics['g_label'])

            elif self.phase == 'fake':
                g_data_epochs[epoch] = (train_metrics['g_data'], train_metrics['g_label'])

            else:
                raise ValueError('unknown training phase %s.' % self.phase)

            #############
            # Validation
            #############
            if 'cls' in self.phase:
                with torch.no_grad():
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    logger.info('val_loss: %s' % this_epoch_val_metric)

                    self._val_loss_tracker.add_metric(this_epoch_val_metric)

                    if self._val_loss_tracker.should_stop_early():
                        logger.info("Ran out of patience. Stopping training.")
                        if self.phase == 'real':
                            # =======================
                            # creating feature space
                            # =======================
                            # load best model
                            best_model_state = self._checkpointer.best_model_state()
                            if best_model_state:
                                self.model.load_state_dict(best_model_state)
                            # get training features
                            for batch in self.data_iterator(self.train_dataset, num_epochs=1, shuffle=False):
                                batch = nn_util.move_to_device(batch, self.cuda_device)
                                features = self.model.feature_extractor(batch['tokens'])
                                for i in range(features.size(0)):
                                    training_features.append({'feature': features[i, :].data.cpu().numpy(),
                                                              'label': batch['label'][i].data.cpu().numpy()})
                            # save validation features
                            v_data = []
                            v_label = []
                            for batch in self.data_iterator(self.validation_dataset, num_epochs=1, shuffle=False):
                                batch = nn_util.move_to_device(batch, self.cuda_device)
                                features = self.model.feature_extractor(batch['tokens'])
                                v_data.append(features.data.cpu().numpy())
                                v_label.extend(batch['label'].data.cpu().numpy())
                            v_data_epochs = (np.vstack(v_data), v_label)
                        break

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

            if 'cls' in self.phase:

                for key, value in val_metrics.items():
                    metrics["validation_" + key] = value

                if self._val_loss_tracker.is_best_so_far():
                    metrics['best_epoch'] = epoch
                    for key, value in val_metrics.items():
                        metrics["best_validation_" + key] = value

                    self._val_loss_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            #############
            # Save model
            #############
            if 'real' in self.phase or 'fake' in self.phase:
                self._save_checkpoint(epoch, self._val_loss_tracker)
            elif 'gan' in self.phase:
                self._save_checkpoint(epoch, self._gan_loss_tracker)
            else:
                raise ValueError('unknown training phase %s.' % self.phase)

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
            'r_data_epochs': r_data_epochs,
            'g_data_epochs': g_data_epochs,
            'v_data_epochs': v_data_epochs,
            'd_loss_epochs': d_loss_epochs,
            'g_loss_epochs': g_loss_epochs,
            'cls_loss_on_real_epochs': cls_loss_on_real_epochs,
            'training_features': training_features
        }
        return meta_data

    def _validation_loss(self) -> Tuple[float, int]:
        logger.info("### Validating ###")
        self.model.eval()

        batches_this_epoch = 0
        val_loss = 0.

        for batch in self.data_iterator(self.validation_dataset, num_epochs=1, shuffle=False):
            batch = nn_util.move_to_device(batch, self.cuda_device)
            features = self.model.feature_extractor(batch['tokens'])
            cls_error = self.model.classifier(features, batch['label'])['loss']

            batches_this_epoch += 1
            val_loss += cls_error.mean().item()

        return val_loss, batches_this_epoch

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
                    ) -> 'GanBertTrainer':

        training_file = params.pop('training_file')
        dev_file = params.pop('dev_file')
        test_file = params.pop('test_file')

        config_file = params.pop('config_file')
        cuda_device = params.pop_int("cuda_device")

        # Data reader
        reader = DatasetReader.from_params(params.pop('dataset_reader'))

        train_dataset = reader.read(cached_path(training_file))
        dev_dataset = reader.read(cached_path(dev_file))
        test_dataset = reader.read(cached_path(test_file))

        noise_reader = DatasetReader.from_params(params.pop("noise_reader"))
        noise_dataset = noise_reader.read("")

        feature_reader = DatasetReader.from_params(params.pop("feature_reader"))
        feature_dataset = feature_reader.read("")

        # Vocabulary
        vocab = Vocabulary.from_instances(train_dataset,
                                          max_vocab_size=config_file.max_vocab_size)
        # logging.info('Vocab size: %s' % vocab.get_vocab_size())
        # vocab.print_statistics()
        # print(vocab.get_token_to_index_vocabulary('labels'))
        # vocab = Vocabulary.from_files(DirConfig.BERT_VOC)
        # vocab.print_statistics()

        # Iterator
        data_iterator = DataIterator.from_params(params.pop("training_iterator"))
        data_iterator.index_with(vocab)
        noise_iterator = DataIterator.from_params(params.pop("noise_iterator"))
        noise_iterator.index_with(vocab)
        feature_iterator = DataIterator.from_params(params.pop("feature_iterator"))
        noise_iterator.index_with(vocab)

        # Model
        feature_extractor = Model.from_params(params.pop("feature_extractor"), vocab=vocab).cuda(cuda_device)
        generator = Model.from_params(params.pop("generator"), vocab=None).cuda(cuda_device)
        discriminator = Model.from_params(params.pop("discriminator"), vocab=None).cuda(cuda_device)
        classifier = Model.from_params(params.pop("classifier")).cuda(cuda_device)

        model = Gan(
            feature_extractor=feature_extractor,
            generator=generator,
            discriminator=discriminator,
            classifier=classifier
        )

        # Optimize
        parameters = []
        for component in [
            model.feature_extractor,
            model.generator,
            model.discriminator,
            model.classifier]:
            parameters += [[n, p] for n, p in component.named_parameters() if p.requires_grad]
        optimizer = GanOptimizer.from_params(parameters, params.pop("optimizer"))

        n_epoch_real = params.pop_int("n_epoch_real")
        n_epoch_gan = params.pop_int("n_epoch_gan")
        n_epoch_fake = params.pop_int("n_epoch_fake")
        batch_size = params.pop_int("batch_size")
        patience = params.pop_int("patience")
        conservative_rate = params.pop_float("conservative_rate")
        num_loop_discriminator = params.pop_int("num_loop_discriminator")
        batch_per_epoch = params.pop_int("batch_per_epoch")
        batch_per_generator = params.pop_int("batch_per_generator")
        gen_step = params.pop_int("gen_step")
        clip_value = params.pop_int("clip_value")
        n_classes = params.pop_int("n_classes")
        phase = params.pop("phase")
        model_real_dir = params.pop("model_real_dir")
        model_gan_dir = params.pop("model_gan_dir")
        model_fake_dir = params.pop("model_fake_dir")
        params.pop("trainer")

        params.assert_empty(__name__)

        return cls(model=model,
                   optimizer=optimizer,

                   train_dataset=train_dataset,
                   validation_dataset=dev_dataset,
                   test_dataset=test_dataset,
                   noise_dataset=noise_dataset,
                   feature_dataset=feature_dataset,

                   data_iterator=data_iterator,
                   noise_iterator=noise_iterator,
                   feature_iterator=feature_iterator,

                   serialization_dir=serialization_dir,

                   n_epoch_real=n_epoch_real,
                   n_epoch_gan=n_epoch_gan,
                   n_epoch_fake=n_epoch_fake,

                   batch_size=batch_size,
                   cuda_device=cuda_device,
                   patience=patience,
                   conservative_rate=conservative_rate,
                   num_loop_discriminator=num_loop_discriminator,
                   batch_per_epoch=batch_per_epoch,
                   batch_per_generator=batch_per_generator,
                   gen_step=gen_step,
                   clip_value=clip_value,
                   n_classes=n_classes,
                   phase=phase,

                   model_real_dir=model_real_dir,
                   model_gan_dir=model_gan_dir,
                   model_fake_dir=model_fake_dir)
