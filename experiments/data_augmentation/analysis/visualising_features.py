from typing import List
import torch
import numpy as np
import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from sklearn.manifold import TSNE

# from config import StanceConfig as ConfigFile
from config import TRECConfig as ConfigFile

random_state = 2019


def visualise(x, epoch, color, is_show=False, is_export=False):
    tsne_model = TSNE(n_components=2,
                      perplexity=50,
                      learning_rate=10,
                      verbose=1,
                      random_state=random_state,
                      init='pca')
    # get coordinates for each tweet
    tsne_points = tsne_model.fit_transform(x)

    plot = bp.figure(plot_width=600, plot_height=600,
                     title='Epoch: %s' % epoch,
                     toolbar_location=None, tools="")

    plot.scatter(x=tsne_points[:, 0],
                 y=tsne_points[:, 1],
                 size=3,
                 color=color)

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=ConfigFile.img_gen_feature_path % epoch)
        print('saved to %s.' % ConfigFile.img_gen_feature_path % epoch)


def over_epoch(epoch_specified: List = None):
    # stance_target = 'a'
    stance_target = 'la'
    # stance_target = 'cc'
    # stance_target = 'la'
    train_meta_data = torch.load(ConfigFile.train_meta_path % stance_target)
    real_train_features = train_meta_data['r_data_epochs'][15]
    real_test_features = torch.load(ConfigFile.test_meta_path % stance_target)

    colors = {
        0: 'blue',
        1: 'red',
        2: 'green'
    }
    if epoch_specified is None:
        epoch_specified = train_meta_data['g_data_epochs'].keys()

    for epoch in epoch_specified:
        gen_features = train_meta_data['g_data_epochs'][epoch]
        labels = [0] * real_train_features.shape[0] + [1] * gen_features.shape[0] + [2] * real_test_features.shape[0]
        visualise(np.concatenate((real_train_features, gen_features, real_test_features)),
                  epoch, colors, labels,
                  False, True)


def visualize_real_features(meta_path, corpus_name, file_frac):
    train_meta_data = torch.load(meta_path % (corpus_name, file_frac))
    n_samples = train_meta_data['r_data_epochs'][0][0].shape[0]

    for epoch in train_meta_data['r_data_epochs'].keys():
        real_train_features = train_meta_data['r_data_epochs'][epoch]
        r_data, r_label = real_train_features
        colors = [bpa.all_palettes['Dark2'][6][label] for label in r_label]
        visualise(r_data,
                  epoch=epoch,
                  color=colors,
                  is_show=False,
                  is_export=True)


def visualize_gen_features(real_meta_path, gan_meta_path, fake_meta_path,
                           corpus_name, file_frac):
    real_meta_data = torch.load(real_meta_path % (corpus_name, file_frac))
    real_train_features = real_meta_data['r_data_epochs'][19][0]

    gan_meta_data = torch.load(gan_meta_path % (corpus_name, file_frac))
    fake_meta_data = torch.load(fake_meta_path % (corpus_name, file_frac))
    palettes = {
        0: 'blue',
        1: 'red',
        2: 'green'
    }

    # for epoch in gan_meta_data['g_data_epochs'].keys():
    #     gen_features = gan_meta_data['g_data_epochs'][epoch]
    #     colors = [palettes[0]] * real_train_features.shape[0] + [palettes[1]] * gen_features.shape[0]
    #     visualise(np.concatenate((real_train_features, gen_features)),
    #               epoch, colors,
    #               False, True)

    for epoch in fake_meta_data['g_data_epochs'].keys():
        fake_features = fake_meta_data['g_data_epochs'][epoch]
        colors = [palettes[0]] * real_train_features.shape[0] + [palettes[1]] * fake_features.shape[0]
        visualise(np.concatenate((real_train_features, fake_features)),
                  epoch, colors,
                  False, True)


if __name__ == '__main__':
    # visualize_real_features(ConfigFile.train_meta_path,
    #                         ConfigFile.corpus_name,
    #                         ConfigFile.hparam['file_frac'])
    visualize_gen_features(ConfigFile.train_real_meta_path,
                           ConfigFile.train_gan_meta_path,
                           ConfigFile.train_fake_meta_path,
                           ConfigFile.corpus_name,
                           ConfigFile.hparam['file_frac'])
