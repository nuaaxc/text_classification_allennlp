from typing import List
import torch
import os
import random
import numpy as np
import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from sklearn.manifold import TSNE

# from config.stance import StanceCfg as cfg
# from config.sst import SSTCfg as cfg
# from config.r8 import R8Cfg as cfg
# from config.offensive import OffensiveCfg as cfg
from config.trec import TRECCfg as cfg

random_state = 2019

# my_palettes = {
#     0: 'red',
#     1: 'blue',
#     2: 'green',
#     3: 'orange',
#     4: 'purple',
#     5: 'DimGrey'
# }

my_palettes = {
    0: 'DimGrey',
    1: 'blue',
    2: 'green',
    3: 'DarkOrange',
    4: 'purple',
    5: 'red'
}


def visualise_debug(x, epoch, fill_color, line_color, alpha, marker,
              is_show=False, is_export=False, output_file=None):
    tsne_model = TSNE(n_components=2,
                      perplexity=50,
                      learning_rate=10,
                      verbose=1,
                      random_state=random_state,
                      init='pca')

    tsne_points = tsne_model.fit_transform(x)

    plot = bp.figure(plot_width=500, plot_height=500,
                     title='Epoch: %s' % epoch,
                     toolbar_location=None, tools="")

    plot.scatter(x=tsne_points[:, 0],
                 y=tsne_points[:, 1],
                 size=5,
                 fill_color=fill_color,
                 line_color=line_color,
                 fill_alpha=alpha,
                 line_alpha=alpha,
                 marker=marker)

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=output_file)
        print('saved to %s.' % output_file)


def visualise(x, epoch, fill_color, line_color, alpha, marker,
              is_show=False, is_export=False, output_file=None):
    tsne_model = TSNE(n_components=2,
                      perplexity=50,
                      learning_rate=10,
                      verbose=1,
                      random_state=random_state,
                      init='pca')

    tsne_points = tsne_model.fit_transform(x)

    plot = bp.figure(plot_width=200, plot_height=200,
                     title='',
                     toolbar_location=None, tools="")
    plot.grid.visible = False
    plot.xaxis.visible = False
    plot.yaxis.visible = False

    plot.scatter(x=tsne_points[:, 0],
                 y=tsne_points[:, 1],
                 size=2,
                 # size=1,
                 fill_color=fill_color,
                 line_color=line_color,
                 fill_alpha=alpha,
                 marker=marker)

    if is_show:
        show(plot)

    if is_export:
        export_png(plot, filename=output_file)
        print('saved to %s.' % output_file)


def visualize_one_group(features1, labels1,
                        is_show,
                        is_export,
                        output_path):
    r_markers = ['circle'] * len(labels1)

    # r_fill_colors = [bpa.all_palettes['Set1'][max(cfg.n_label, 3)][int(label)] for label in labels1]
    # r_line_colors = [bpa.all_palettes['Set1'][max(cfg.n_label, 3)][int(label)] for label in labels1]

    r_fill_colors = [my_palettes[int(label)] for label in labels1]
    r_line_colors = [my_palettes[int(label)] for label in labels1]

    r_alphas = [0.5] * len(labels1)

    visualise(features1,
              '',
              fill_color=r_fill_colors,
              line_color=r_line_colors,
              alpha=r_alphas,
              marker=r_markers,
              is_show=is_show,
              is_export=is_export,
              output_file=output_path)


def visualize_two_groups(features1, labels1,
                         features3, labels3,
                         is_show,
                         is_export,
                         output_path):
    r_markers = ['circle'] * len(labels1)
    v_markers = ['triangle'] * len(labels3)

    r_fill_colors = [bpa.all_palettes['Dark2'][max(cfg.n_label, 3)][label] for label in labels1]
    v_fill_colors = ['white'] * len(labels3)

    r_line_colors = ['white'] * len(labels1)
    v_line_colors = [bpa.all_palettes['Dark2'][max(cfg.n_label, 3)][int(label)] for label in labels3]

    r_alphas = [0.5] * len(labels1)
    v_alphas = [1.] * len(labels3)

    visualise(np.concatenate((features1, features3)),
              '',
              fill_color=r_fill_colors + v_fill_colors,
              line_color=r_line_colors + v_line_colors,
              alpha=r_alphas + v_alphas,
              marker=r_markers + v_markers,
              is_show=is_show,
              is_export=is_export,
              output_file=output_path)


def visualize_three_groups(features1, labels1,
                           features2, labels2,
                           features3, labels3,
                           is_show,
                           is_export,
                           output_path):
    r_markers = ['circle'] * len(labels1)
    t_markers = ['diamond'] * len(labels2)
    v_markers = ['triangle'] * len(labels3)

    r_fill_colors = [bpa.all_palettes['Dark2'][max(cfg.n_label, 3)][label] for label in labels1]
    # t_fill_colors = ['blue'] * len(t_labels)
    t_fill_colors = [bpa.all_palettes['Dark2'][max(cfg.n_label, 3)][label] for label in labels2]
    v_fill_colors = ['white'] * len(labels3)

    r_line_colors = ['white'] * len(labels1)
    t_line_colors = ['black'] * len(labels2)
    v_line_colors = [bpa.all_palettes['Dark2'][max(cfg.n_label, 3)][int(label)] for label in labels3]

    r_alphas = [0.5] * len(labels1)
    t_alphas = [1.] * len(labels2)
    v_alphas = [1.] * len(labels3)

    visualise(np.concatenate((features1, features2, features3)),
              '',
              fill_color=r_fill_colors + t_fill_colors + v_fill_colors,
              line_color=r_line_colors + t_line_colors + v_line_colors,
              alpha=r_alphas + t_alphas + v_alphas,
              marker=r_markers + t_markers + v_markers,
              is_show=is_show,
              is_export=is_export,
              output_file=output_path)


def visualize_real_features_train(input_path, output_path, is_show=False, is_export=False):
    """
    Visualization of real train/validation/test features
    """
    features = torch.load(input_path)
    train_features = features['train_features']
    train_labels = features['train_labels']

    visualize_one_group(train_features, train_labels,
                        is_show,
                        is_export,
                        output_path)


def visualize_real_features_train_dev_test(input_path, output_path, is_show=False, is_export=False):
    """
    Visualization of real train/validation/test features
    """
    features = torch.load(input_path)
    train_features = features['train_features']
    train_labels = features['train_labels']
    validation_features = features['validation_features']
    validation_labels = features['validation_labels']
    test_features = features['test_features']
    test_labels = features['test_labels']

    visualize_three_groups(train_features, train_labels,
                           test_features, test_labels,
                           validation_features, validation_labels,
                           is_show,
                           is_export,
                           output_path)


def visualize_gen_features(input_gen_path, output_path, is_show=False, is_export=False):
    gen_data = torch.load(input_gen_path)
    gen_features = np.array(gen_data['gen_features'])
    gen_labels = np.array(gen_data['gen_labels'])
    print('# gen labels:', len(gen_labels))
    sample_size = min(len(gen_labels),
                      cfg.hp.training_size[cfg.hp.file_ratio] * (cfg.hp.gen_step - 1),
                      10000)
    print('sample_size:', sample_size)
    index_sample = random.sample(range(len(gen_labels)), sample_size)
    gen_features_sample = gen_features[index_sample]
    gen_labels_sample = gen_labels[index_sample]

    visualize_one_group(gen_features_sample, gen_labels_sample,
                        is_show,
                        is_export,
                        output_path)


def visualize_fake_features(input_train_path, input_gen_path, output_path, is_show=False, is_export=False):
    features = torch.load(input_train_path)
    train_features = features['train_features']
    train_labels = features['train_labels']
    print('# train labels:', len(train_labels))

    test_features = features['test_features']
    test_labels = features['test_labels']
    print('# test labels:', len(test_labels))

    gen_data = torch.load(input_gen_path)
    gen_features = np.array(gen_data['gen_features'])
    gen_labels = np.array(gen_data['gen_labels'])
    print('# gen labels:', len(gen_labels))
    sample_size = min(len(gen_labels), 2 * len(train_labels), 5000)
    index_sample = random.sample(range(len(gen_labels)), sample_size)
    gen_features_sample = gen_features[index_sample]
    gen_labels_sample = gen_labels[index_sample]

    # visualize_three_groups(train_features, train_labels,
    #                        test_features, test_labels,
    #                        gen_features_sample, gen_labels_sample,
    #                        is_show,
    #                        is_export,
    #                        output_path)

    visualize_two_groups(train_features, train_labels,
                         gen_features_sample, gen_labels_sample,
                         is_show,
                         is_export,
                         output_path)


def visualize_gen_features_old(real_meta_path, gan_meta_path, test_meta_path,
                               corpus_name, file_ratio):
    real_meta_data = torch.load(real_meta_path % (corpus_name, file_ratio))
    real_train_features, real_training_labels = \
        real_meta_data['r_data_epochs'][real_meta_data['metrics']['best_epoch']]
    test_meta_data = torch.load(test_meta_path % (corpus_name, file_ratio))

    # v_data, v_labels = real_meta_data['v_data_epochs']
    t_data, t_labels = test_meta_data['r_data']

    gan_meta_data = torch.load(gan_meta_path % (corpus_name, file_ratio))
    print('total epochs:', len(gan_meta_data['g_data_epochs'].keys()))

    img_dir = os.path.join(cfg.result_dir,
                           '_'.join(['img',
                                     'r', str(cfg.HP.file_ratio)
                                     ]))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for epoch in gan_meta_data['g_data_epochs'].keys():
        gen_features, gen_labels = gan_meta_data['g_data_epochs'][epoch]

        r_markers = ['circle'] * len(real_training_labels)
        t_markers = ['diamond'] * len(t_labels)
        g_markers = ['triangle'] * len(gen_labels)

        r_fill_colors = [bpa.all_palettes['Dark2'][cfg.n_label][label] for label in real_training_labels]
        # t_fill_colors = ['blue'] * len(t_labels)
        t_fill_colors = [bpa.all_palettes['Dark2'][cfg.n_label][label] for label in t_labels]
        g_fill_colors = ['white'] * len(gen_labels)

        r_line_colors = ['white'] * len(real_training_labels)
        t_line_colors = ['black'] * len(t_labels)
        g_line_colors = [bpa.all_palettes['Dark2'][cfg.n_label][int(label)] for label in gen_labels]

        r_alphas = [0.5] * len(real_training_labels)
        t_alphas = [1.] * len(t_labels)
        g_alphas = [1.] * len(gen_labels)

        output_file = os.path.join(img_dir, cfg.img_gen_feature_path % epoch)

        visualise(np.concatenate((real_train_features, t_data, gen_features)),
                  epoch,
                  fill_color=r_fill_colors + t_fill_colors + g_fill_colors,
                  line_color=r_line_colors + t_line_colors + g_line_colors,
                  alpha=r_alphas + t_alphas + g_alphas,
                  marker=r_markers + t_markers + g_markers,
                  is_show=False,
                  is_export=True,
                  output_file=output_file)


if __name__ == '__main__':

    img_dir = os.path.join(cfg.result_dir, '_'.join(['img', 'r', str(cfg.hp.file_ratio)]))
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # visualize_real_features_train(input_path=cfg.train_real_meta_path % (cfg.corpus_name, cfg.hp.file_ratio),
    #                               output_path=os.path.join(img_dir, cfg.img_real_feature_path %
    #                                                        (cfg.corpus_name, int(cfg.hp.file_ratio * 100))),
    #                               is_show=True,
    #                               is_export=True)

    visualize_gen_features(cfg.train_fake_meta_path % (cfg.corpus_name, cfg.hp.file_ratio),
                           output_path=os.path.join(img_dir, cfg.img_fake_feature_path %
                                                    (cfg.corpus_name, int(cfg.hp.file_ratio * 100))),
                           is_show=True,
                           is_export=True)

    # visualize_fake_features(cfg.train_real_meta_path % (cfg.corpus_name, cfg.hp.file_ratio),
    #                         cfg.train_fake_meta_path % (cfg.corpus_name, cfg.hp.file_ratio),
    #                         output_path=os.path.join(img_dir, cfg.img_fake_feature_path),
    #                         is_show=True,
    #                         is_export=True)
