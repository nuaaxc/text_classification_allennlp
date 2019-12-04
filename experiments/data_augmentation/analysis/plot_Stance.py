import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from bokeh.models import Legend
from bokeh.layouts import column

from config.stance import *

results = {
    5: [40.54, 42.88, 43.93, 45.33, 43.13, 47.46, 46.94, 46.51],
    20: [52.40, 54.97, 54.11, 55.15, 57.20, 57.07, 57.09, 56.58],
    50: [56.16, 56.46, 56.54, 56.65, 56.74, 57.24, 57.12, 58.11],
    100: [60.86, 62.16, 62.25, 61.99, 62.53, 62.59, 61.96, 61.57]
}


def performance_plot(is_export=False):
    x_axis = [-1, 0, 1, 2, 3, 4, 5, 6]
    plot = bp.figure(plot_width=200, plot_height=250,
                     title='Stance (N=4,163)',
                     toolbar_location=None, tools="")
    plot.title.align = 'center'
    plot.title.text_font_size = '15pt'
    plot.yaxis.major_label_text_font_size = '11pt'
    plot.yaxis.major_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_size = '11pt'
    plot.xaxis.major_label_text_font_style = 'bold'

    plot.line(x_axis, results[5], color='red', line_width=2)
    plot.square(x_axis, results[5], size=10, fill_color=None, line_color='red', line_width=1)

    plot.line(x_axis, results[20], color='blue', line_width=2)
    plot.hex(x_axis, results[20], size=10, fill_color=None, line_color='blue', line_width=1)

    plot.line(x_axis, results[50], color='black', line_width=2)
    plot.circle(x_axis, results[50], size=10, fill_color=None, line_color='black', line_width=1)

    plot.line(x_axis, results[100], color='purple', line_width=2)
    plot.triangle(x_axis, results[100], size=10, fill_color=None, line_color='purple', line_width=1)

    plot.xaxis.major_label_overrides = {-1: 'N'}

    show(plot)

    if is_export:
        export_png(plot, filename=StanceCfg.img_quant_path)
        print('saved to %s.' % StanceCfg.img_quant_path)


if __name__ == '__main__':
    performance_plot(is_export=True)
    # performance_plot(is_export=False)
