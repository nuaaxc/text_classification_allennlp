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


def visualise(x, epoch, fill_color, line_color, alpha, marker, is_show=False, is_export=False):
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
                 size=10,
                 fill_color=fill_color,
                 line_color=line_color,
                 fill_alpha=alpha,
                 line_alpha=alpha,
                 marker=marker)

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=ConfigFile.img_gen_feature_path % epoch)
        print('saved to %s.' % ConfigFile.img_gen_feature_path % epoch)


def visualize_features(real_meta_path, gan_meta_path, test_meta_path,
                       corpus_name, file_frac):
    real_meta_data = torch.load(real_meta_path % (corpus_name, file_frac))
    real_train_features, real_training_labels = \
        real_meta_data['r_data_epochs'][real_meta_data['metrics']['best_epoch']]
    test_meta_data = torch.load(test_meta_path % (corpus_name, file_frac))

    # v_data, v_labels = real_meta_data['v_data_epochs']
    t_data, t_labels = test_meta_data['r_data']

    gan_meta_data = torch.load(gan_meta_path % (corpus_name, file_frac))
    for epoch in gan_meta_data['g_data_epochs'].keys():
        gen_features, gen_labels = gan_meta_data['g_data_epochs'][epoch]

        r_markers = ['circle'] * len(real_training_labels)
        t_markers = ['diamond'] * len(t_labels)
        g_markers = ['triangle'] * len(gen_labels)

        r_fill_colors = [bpa.all_palettes['Dark2'][6][label] for label in real_training_labels]
        # t_fill_colors = ['blue'] * len(t_labels)
        t_fill_colors = [bpa.all_palettes['Dark2'][6][label] for label in t_labels]
        g_fill_colors = ['white'] * len(gen_labels)

        r_line_colors = ['white'] * len(real_training_labels)
        t_line_colors = ['black'] * len(t_labels)
        g_line_colors = [bpa.all_palettes['Dark2'][6][int(label)] for label in gen_labels]

        r_alphas = [0.5] * len(real_training_labels)
        t_alphas = [1.] * len(t_labels)
        g_alphas = [1.] * len(gen_labels)

        visualise(np.concatenate((real_train_features, t_data, gen_features)),
                  epoch,
                  fill_color=r_fill_colors + t_fill_colors + g_fill_colors,
                  line_color=r_line_colors + t_line_colors + g_line_colors,
                  alpha=r_alphas + t_alphas + g_alphas,
                  marker=r_markers + t_markers + g_markers,
                  is_show=False,
                  is_export=True)


if __name__ == '__main__':
    # visualize_features(ConfigFile.train_real_meta_path,
    #                    ConfigFile.train_gan_meta_path,
    #                    ConfigFile.test_meta_path,
    #                    ConfigFile.corpus_name,
    #                    ConfigFile.hparam['file_frac'])
    visualize_features(ConfigFile.train_real_meta_path,
                       ConfigFile.train_fake_meta_path,
                       ConfigFile.test_meta_path,
                       ConfigFile.corpus_name,
                       ConfigFile.hparam['file_frac'])
