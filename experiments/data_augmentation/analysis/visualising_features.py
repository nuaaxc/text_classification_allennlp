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
                 size=2,
                 color=[color[label] for label in labels])

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=ConfigFile.img_gen_feature_path % epoch)
        print('saved to %s.' % ConfigFile.img_gen_feature_path % epoch)


if __name__ == '__main__':
    stance_target = 'a'
    train_meta_data = torch.load(ConfigFile.train_meta_path % stance_target)
    real_features = train_meta_data['r_data_epochs'][19]

    colors = {
        0: 'red',
        1: 'blue'
    }

    for epoch in train_meta_data['g_data_epochs'].keys():
        gen_features = train_meta_data['g_data_epochs'][epoch]
        labels = [0] * real_features.shape[0] + [1] * gen_features.shape[0]
        visualise(np.concatenate((real_features, gen_features)),
                  epoch, colors, labels,
                  False, True)
