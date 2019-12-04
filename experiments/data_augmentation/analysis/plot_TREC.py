import bokeh.plotting as bp
import bokeh.palettes as bpa
from bokeh.io import show, export_png
from bokeh.models import Legend
from bokeh.layouts import column

from config.trec import *

results = {
    5: [92, 92.44, 93.22, 92.7, 92.18, 92.18, 92.18, 92.13],
    20: [94.8, 95.16, 95.41, 95.62, 95.41, 96.09, 95.36, 95.36],
    50: [95.2, 95.31, 96.09, 95.83, 96.61, 96.09, 96.35, 96.09],
    100: [96.6, 96.87, 97.13, 96.87, 97.13, 97.26, 98.04, 96.87]
}


def performance_plot(is_export=False):
    x_axis = [-1, 0, 1, 2, 3, 4, 5, 6]
    plot = bp.figure(plot_width=300, plot_height=250,
                     title='TREC (N=5,452)',
                     toolbar_location=None, tools="")
    plot.title.align = 'center'
    plot.title.text_font_size = '15pt'
    plot.yaxis.major_label_text_font_size = '11pt'
    plot.yaxis.major_label_text_font_style = 'bold'
    plot.xaxis.major_label_text_font_size = '11pt'
    plot.xaxis.major_label_text_font_style = 'bold'
    legend_it = []

    c1 = plot.line(x_axis, results[5], color='red', line_width=2)
    m1 = plot.square(x_axis, results[5], size=10, fill_color=None, line_color='red', line_width=1)

    c2 = plot.line(x_axis, results[20], color='blue', line_width=2)
    m2 = plot.hex(x_axis, results[20], size=10, fill_color=None, line_color='blue', line_width=1)

    c3 = plot.line(x_axis, results[50], color='black', line_width=2)
    m3 = plot.circle(x_axis, results[50], size=10, fill_color=None, line_color='black', line_width=1)

    c4 = plot.line(x_axis, results[100], color='purple', line_width=2)
    m4 = plot.triangle(x_axis, results[100], size=10, fill_color=None, line_color='purple', line_width=1)

    legend_it.append(('5%', [c1, m1]))
    legend_it.append(('20%', [c2, m2]))
    legend_it.append(('50%', [c3, m3]))
    legend_it.append(('100%', [c4, m4]))
    plot.xaxis.major_label_overrides = {-1: 'N'}
    legend = Legend(items=legend_it, location=(0, 0))
    legend.click_policy = "mute"
    legend.label_text_font_size = '14pt'
    plot.add_layout(legend, 'right')
    show(plot)

    if is_export:
        export_png(plot, filename=TRECCfg.img_quant_path)
        print('saved to %s.' % TRECCfg.img_quant_path)


if __name__ == '__main__':
    performance_plot(is_export=True)
    # performance_plot(is_export=False)
