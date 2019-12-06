import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from bokeh.models import Legend
from bokeh.layouts import column

from config.sst import *
from experiments.data_augmentation.analysis.utils import performance_gain


results = {
    5: [83.52, 83.84, 83.79, 83.84, 84.53, 84.65, 84.15, 83.98],
    20: [87.05, 87.98, 88.09, 88.09, 87.91, 87.97, 87.87, 87.76],
    50: [89.11, 89.94, 89.89, 90.00, 90.17, 90.44, 90.72, 90.61],
    100: [91.01, 91.46, 91.57, 91.47, 91.37, 91.86, 91.40, 91.35]
}


def performance_plot(is_export=False):
    x_axis = [-1, 0, 1, 2, 3, 4, 5, 6]
    plot = bp.figure(plot_width=220, plot_height=200,
                     title='SST-2 (N=7,447)',
                     toolbar_location=None, tools="")
    plot.title.align = 'center'
    plot.title.text_font_size = '15pt'
    plot.yaxis.major_label_text_font_size = '11pt'
    plot.yaxis.major_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_size = '11pt'
    plot.xaxis.major_label_text_font_style = 'bold'
    plot.yaxis.axis_label = 'Accuracy'
    plot.yaxis.axis_label_text_font_style = 'bold'
    plot.yaxis.axis_label_text_font_size = '13pt'

    plot.line(x_axis, results[5], color='red', line_width=2)
    plot.square(x_axis, results[5], size=10, fill_color=None, line_color='red', line_width=1)

    plot.line(x_axis, results[20], color='blue', line_width=2)
    plot.hex(x_axis, results[20], size=10, fill_color=None, line_color='blue', line_width=1)

    plot.line(x_axis, results[50], color='black', line_width=2)
    plot.circle(x_axis, results[50], size=10, fill_color=None, line_color='black', line_width=1)

    plot.line(x_axis, results[100], color='purple', line_width=2)
    plot.triangle(x_axis, results[100], size=10, fill_color=None, line_color='purple', line_width=1)

    plot.xaxis.major_label_overrides = {-1: 'R'}

    show(plot)

    if is_export:
        export_png(plot, filename=SSTCfg.img_quant_path)
        print('saved to %s.' % SSTCfg.img_quant_path)


if __name__ == '__main__':
    performance_plot(is_export=True)
    # performance_plot(is_export=False)
    # performance_gain(results)
