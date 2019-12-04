import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from bokeh.models import Legend
from bokeh.layouts import column
from bokeh.models import NumeralTickFormatter

from config.r8 import *

results = {
    5: [89.55, 89.98, 90.02, 89.98, 90.02, 90.07, 89.98, 89.93],
    20: [95.45, 95.83, 95.81, 95.77, 95.95, 95.90, 95.90, 95.81],
    50: [96.89, 96.96, 96.97, 96.92, 97.01, 97.07, 96.92, 96.96],
    100: [96.93, 96.96, 97.10, 97.07, 96.92, 96.96, 97.01, 96.92],
}


def performance_plot(is_export=False):
    x_axis = [-1, 0, 1, 2, 3, 4, 5, 6]

    # P1
    p1 = bp.figure(
        plot_width=200, plot_height=100,
                   title='R8 (N=7,674)', toolbar_location=None, tools="",
                   y_range=(96.8, 97.2)
    )
    p1.title.align = 'center'
    p1.title.text_font_size = '14pt'
    p1.xaxis.visible = False
    p1.yaxis.major_label_text_font_size = '11pt'
    p1.yaxis.major_label_text_font_style = 'bold'
    p1.yaxis.ticker = [96.8, 96.9, 97.0, 97.1, 97.2]
    p1.yaxis.formatter = NumeralTickFormatter(format='0.0')
    # p1.yaxis.ticker = [96.89, 97.01, 97.11]
    p1.line(x_axis, results[50], color='black', line_width=2)
    p1.circle(x_axis, results[50], size=10, fill_color=None, line_color='black', line_width=1)
    p1.line(x_axis, results[100], color='purple', line_width=2)
    p1.triangle(x_axis, results[100], size=10, fill_color=None, line_color='purple', line_width=1)

    # P2
    p2 = bp.figure(
        plot_width=200, plot_height=65,
                   title='', toolbar_location=None, tools='',
                   y_range=(95.30, 96.10))
    p2.xaxis.visible = False
    p2.yaxis.ticker = [95.4, 95.6, 95.8, 96]
    p2.yaxis.formatter = NumeralTickFormatter(format='0.0')
    p2.yaxis.major_label_text_font_size = '11pt'
    p2.yaxis.major_label_text_font_style = 'bold'
    p2.line(x_axis, results[20], color='blue', line_width=2)
    p2.hex(x_axis, results[20], size=10, fill_color=None, line_color='blue', line_width=1)

    # P3
    p3 = bp.figure(
        plot_width=200, plot_height=85,
                   title='', toolbar_location=None, tools="",
                   y_range=(89.4, 90.20))
    p3.yaxis.ticker = [89.4, 89.6, 89.8, 90.0]
    p3.yaxis.formatter = NumeralTickFormatter(format='0.0')
    p3.yaxis.major_label_text_font_size = '11pt'
    p3.yaxis.major_label_text_font_style = 'bold'
    p3.xaxis.major_label_text_font_size = '11pt'
    p3.xaxis.major_label_text_font_style = 'bold'
    p3.line(x_axis, results[5], color='red', line_width=2)
    p3.square(x_axis, results[5], size=7, fill_color=None, line_color='red', line_width=1)

    p3.xaxis.major_label_overrides = {-1: 'N'}

    show(column(p1, p2, p3))

    if is_export:
        export_png(column(p1, p2, p3),
                   filename=R8Cfg.img_quant_path)
        print('saved to %s.' % R8Cfg.img_quant_path)


if __name__ == '__main__':
    # performance_plot(is_export=False)
    performance_plot(is_export=True)
