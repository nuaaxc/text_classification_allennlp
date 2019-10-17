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


def visualise(x, epoch, fill_color, line_color, alpha, markers, is_show=False, is_export=False):
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
                 size=8,
                 fill_color=fill_color,
                 line_color=line_color,
                 fill_alpha=alpha,
                 line_alpha=alpha,
                 marker=markers)

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=ConfigFile.img_gen_feature_path % epoch)
        print('saved to %s.' % ConfigFile.img_gen_feature_path % epoch)


def visualize_features(real_meta_path, gan_meta_path, fake_meta_path,
                       corpus_name, file_frac):
    real_meta_data = torch.load(real_meta_path % (corpus_name, file_frac))
    real_train_features, real_training_labels = \
        real_meta_data['r_data_epochs'][real_meta_data['metrics']['best_epoch']]

    palettes = {
        0: 'blue',
        1: 'red',
        2: 'green'
    }

    gan_meta_data = torch.load(gan_meta_path % (corpus_name, file_frac))
    for epoch in gan_meta_data['g_data_epochs'].keys():
        gen_features, gen_labels = gan_meta_data['g_data_epochs'][epoch]

        r_markers = ['circle'] * len(real_training_labels)
        g_markers = ['triangle'] * len(gen_labels)
        markers = r_markers + g_markers

        r_fill_colors = [bpa.all_palettes['Dark2'][6][label] for label in real_training_labels]
        g_fill_colors = ['white'] * len(gen_labels)
        fill_colors = r_fill_colors + g_fill_colors

        r_line_colors = ['grey'] * len(real_training_labels)
        g_line_colors = [bpa.all_palettes['Dark2'][6][int(label)] for label in gen_labels]
        line_colors = r_line_colors + g_line_colors

        r_alphas = [1.] * len(real_training_labels)
        g_alphas = [1.] * len(gen_labels)
        alphas = r_alphas + g_alphas
        visualise(np.concatenate((real_train_features, gen_features)),
                  epoch, fill_colors, line_colors, alphas, markers,
                  False, True)

    # fake_meta_data = torch.load(fake_meta_path % (corpus_name, file_frac))
    # for epoch in fake_meta_data['g_data_epochs'].keys():
    #     fake_features = fake_meta_data['g_data_epochs'][epoch]
    #     colors = [palettes[0]] * real_train_features.shape[0] + [palettes[1]] * fake_features.shape[0]
    #     visualise(np.concatenate((real_train_features, fake_features)),
    #               epoch, colors,
    #               False, True)


if __name__ == '__main__':
    visualize_features(ConfigFile.train_real_meta_path,
                       ConfigFile.train_gan_meta_path,
                       ConfigFile.train_fake_meta_path,
                       ConfigFile.corpus_name,
                       ConfigFile.hparam['file_frac'])
