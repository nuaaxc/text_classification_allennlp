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


def visualise(x, epoch, color, labels, is_show=False, is_export=False):
    tsne_model = TSNE(n_components=2,
                      perplexity=100,
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
                 size=10,
                 color=[color[label] for label in labels])

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=ConfigFile.img_gen_feature_path % epoch)
        print('saved to %s.' % ConfigFile.img_gen_feature_path % epoch)


def over_epoch(epoch_specified: List =None):
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


def visualize_real_features():
    train_meta_data = torch.load(ConfigFile.train_meta_path)
    real_train_features = train_meta_data['r_data_epochs'][0]
    print(real_train_features)
    exit()
    print(real_train_features.shape)

    labels = [0] * real_train_features.shape[0]

    for epoch in train_meta_data['r_data_epochs'].keys():
        real_train_features = train_meta_data['r_data_epochs'][epoch]
        print(real_train_features)

        visualise(real_train_features,
                  epoch=epoch,
                  color=bpa.all_palettes['Accent'][6],
                  labels=labels,
                  is_show=False,
                  is_export=True)


if __name__ == '__main__':
    # over_epoch(list(range(451, 461)))
    visualize_real_features()
