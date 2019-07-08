from typing import List
import torch
import numpy as np
import bokeh.plotting as bp
from bokeh.io import show, export_png
from sklearn.manifold import TSNE

from config import StanceConfig as ConfigFile

random_state = 2019


def visualise(x, epoch, color, labels, is_show=False, is_export=False):
    tsne_model = TSNE(n_components=2,
                      perplexity=50,
                      learning_rate=10,
                      verbose=1,
                      random_state=random_state,
                      init='pca')
    # get coordinates for each tweet
    tsne_points = tsne_model.fit_transform(x)

    plot = bp.figure(plot_width=300, plot_height=300,
                     title='Epoch: %s' % epoch,
                     toolbar_location=None, tools="")

    plot.scatter(x=tsne_points[:, 0],
                 y=tsne_points[:, 1],
                 size=1,
                 color=[color[label] for label in labels])

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=ConfigFile.img_gen_feature_path % epoch)
        print('saved to %s.' % ConfigFile.img_gen_feature_path % epoch)


def over_epoch(epoch_specified: List =None):
    # stance_target = 'a'
    # stance_target = 'cc'
    stance_target = 'la'
    train_meta_data = torch.load(ConfigFile.train_meta_path % stance_target)
    real_features = train_meta_data['r_data_epochs'][19]

    colors = {
        0: 'red',
        1: 'blue'
    }
    if epoch_specified is None:
        epoch_specified = train_meta_data['g_data_epochs'].keys()

    for epoch in epoch_specified:
        gen_features = train_meta_data['g_data_epochs'][epoch]
        labels = [0] * real_features.shape[0] + [1] * gen_features.shape[0]
        visualise(np.concatenate((real_features, gen_features)),
                  epoch, colors, labels,
                  False, True)


if __name__ == '__main__':
    over_epoch(list(range(451, 470)))
