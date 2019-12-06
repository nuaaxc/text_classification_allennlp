import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from bokeh.models import Legend
from bokeh.layouts import column

from config.offensive import *
from experiments.data_augmentation.analysis.utils import performance_gain

results = {
    10: [67.38, 68.9, 69, 70.89, 70.54, 70.15, 68.84, 67.58],
    20: [74.11, 75.11, 74.63, 75.03, 74.63, 75.85, 75.59, 75.58],
    50: [76.66, 77.72, 79.09, 78.37, 77.31, 77.7, 78.47, 78.33],
    100: [78.27, 78.87, 78.97, 79.78, 80.60, 79.16, 78.66, 78.35]
}


def performance_plot(is_export=False):
    x_axis = [-1, 0, 1, 2, 3, 4, 5, 6]
    plot = bp.figure(plot_width=220, plot_height=200,
                     title='OE (N=14,100)',
                     toolbar_location=None, tools='', y_axis_type='log')
    plot.title.align = 'center'
    plot.title.text_font_size = '15pt'
    plot.yaxis.major_label_text_font_size = '11pt'
    plot.yaxis.major_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_size = '11pt'
    plot.xaxis.major_label_text_font_style = 'bold'
    plot.yaxis.axis_label = 'F1-Macro'
    plot.yaxis.axis_label_text_font_style = 'bold'
    plot.yaxis.axis_label_text_font_size = '13pt'

    plot.line(x_axis, results[10], color='red', line_width=2)
    plot.square(x_axis, results[10], size=10, fill_color=None, line_color='red', line_width=1)

    plot.line(x_axis, results[20], color='blue', line_width=2)
    plot.hex(x_axis, results[20], size=10, fill_color=None, line_color='blue', line_width=1)

    plot.line(x_axis, results[50], color='black', line_width=2)
    plot.circle(x_axis, results[50], size=10, fill_color=None, line_color='black', line_width=1)

    plot.line(x_axis, results[100], color='purple', line_width=2)
    plot.triangle(x_axis, results[100], size=10, fill_color=None, line_color='purple', line_width=1)

    plot.xaxis.major_label_overrides = {-1: 'R'}
    # plot.xaxis.major_label_overrides = {-1: 'R', 0: '1', 1: '2', 2: '4', 3: '8', 4: '16', 5: '32', 6: '64'}

    show(plot)

    if is_export:
        export_png(plot, filename=OffensiveCfg.img_quant_path)
        print('saved to %s.' % OffensiveCfg.img_quant_path)


if __name__ == '__main__':
    performance_plot(is_export=True)
    # performance_plot(is_export=False)
    # performance_gain(results)

