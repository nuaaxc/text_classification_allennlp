import torch
import numpy as np
import bokeh.plotting as bp
from bokeh.io import show, export_png
from sklearn.manifold import TSNE

from config import StanceConfig

random_state = 2019


def visualise(x, color, labels):
    tsne_model = TSNE(n_components=2,
                      perplexity=50,
                      learning_rate=10,
                      verbose=1,
                      random_state=random_state,
                      init='pca')
    # get coordinates for each tweet
    tsne_points = tsne_model.fit_transform(x)

    plot = bp.figure(plot_width=300, plot_height=300,
                     # title=DirConfig.model_name_dict[model_name],
                     toolbar_location=None, tools="")

    plot.scatter(x=tsne_points[:, 0],
                 y=tsne_points[:, 1],
                 size=2,
                 color=[color[label] for label in labels])

    show(plot)


if __name__ == '__main__':
    config_file = StanceConfig
    stance_target = 'a'
    real_features = torch.load(config_file.real_data_path % stance_target)[17]
    gen_features = torch.load(config_file.gen_data_path % stance_target)[17]

    colors = {
        0: 'red',
        1: 'blue'
    }

    labels = [0] * real_features.shape[0] + \
             [1] * gen_features.shape[0]
    visualise(np.concatenate((real_features, gen_features)),
              colors,
              labels)
