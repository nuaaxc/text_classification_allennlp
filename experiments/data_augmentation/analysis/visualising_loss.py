import torch
import numpy as np
import bokeh.plotting as bp
from bokeh.io import show, export_png

from config import StanceConfig

if __name__ == '__main__':
    config_file = StanceConfig
    stance_target = 'a'
    train_meta_data = torch.load(config_file.train_meta_path % stance_target)

    d_loss_epochs = train_meta_data['d_loss_epochs']
    g_loss_epochs = train_meta_data['g_loss_epochs']

    plot = bp.figure(plot_width=300, plot_height=300,
                     # title=DirConfig.model_name_dict[model_name],
                     toolbar_location=None, tools="")

    plot.line(list(d_loss_epochs.keys()), list(d_loss_epochs.values()),
              line_width=2,
              line_color='red',
              legend='d_loss')
    plot.line(list(g_loss_epochs.keys()), list(g_loss_epochs.values()),
              line_width=2,
              line_color='blue',
              legend='g_loss')
    show(plot)
