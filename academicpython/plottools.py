# -*- coding: utf-8 -*-
"""plottools.py
Tools for plotting with matplotlib. Use along with scienceplot.
Author: Chong-Chong He

"""

import os
import matplotlib
import matplotlib.pyplot as plt

PLOT_DIR = '.'
SAVING = True

def set_plotdir(path):
    """
    Set the default directory where plots are saved. At the same time,
    """

    if path[-1]=="/":
        path = path[:-1]
    global PLOT_DIR
    PLOT_DIR = path
    if not os.path.isdir(path):
        os.makedirs(path, )

def set_save(saving=True):
    """
    Specify whether plots should be saved.
    This can turn off saving plots when testing.
    """
    global SAVING
    SAVING = saving

def clean_sharex(axes=None, hspace=None):
    """
    For figure with multiple axes that share the x-axis, only
    keep the tick labels for the lower one, and remove the vertical
    space between the axes.
    """
    if axes is None:
        axes = plt.gcf().axes
    # subplots_adjust(hspace=0.001)
    if hspace is not None:
        plt.subplots_adjust(hspace=hspace)
    if axes.ndim == 1:
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel('')
    elif axes.ndim == 2:
        for axrow in axes[:-1]:
            for ax in axrow:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_xlabel('')

def clean_sharey(axes=None):
    """
    For figure with multiple axes that share the y-axis, only
    keep the tick labels for the left one, and remove the horizontal
    space between the axes.
    """
    if axes is None:
        axes = plt.gcf().axes
    plt.subplots_adjust(wspace=0.02)
    # if axes.ndim == 1:
    if np.ndim(axes) == 1:
        for ax in axes[1:]:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel('')
    # elif axes.ndim == 2:
    elif np.ndim(axes) == 2:
        for axcol in axes:
            for ax in axcol[1:]:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_ylabel('')

def sized_figure(rows=1, columns=1, mergex=True, mergey=True,
                     rescale=1.0, top=False, right=False,
                     ret_size=False, figsize=None,
                     **kwargs):

    ratio = 0.68
    w_inch = 3.6 * rescale
    h_inch = w_inch * ratio
    l_inch = 0.68
    r_inch = 0.14 if not right else l_inch
    b_inch = 0.532
    t_inch = 0.14 if not top else b_inch
    fx = columns * w_inch + l_inch + r_inch
    fy = rows * h_inch + b_inch + t_inch
    if not mergex:
        fy += (rows - 1) * (b_inch + t_inch)
    if ret_size:
        return [fx, fy]
    if figsize is None:
        figsize = [fx, fy]
    f, ax = plt.subplots(rows, columns, figsize=figsize, **kwargs)
    plt.subplots_adjust(left=l_inch/fx, right=1-r_inch/fx,
                    bottom=b_inch/fy, top=1-t_inch/fy,
                    wspace=0,
                    hspace=0 if mergex else (b_inch + t_inch) / h_inch,
    )
    return f, ax

def save_pdfpng(filename, filedir='.', fig=None, dpi=None, **kwargs):
    """
    Save the current figure to PDF and PNG in PLOT_DIR.
    PLOT_DIR and TAG are used
    """

    if not SAVING:
        return
    is_png = 1
    if len(filename) > 4:
        if filename[-4:] == ".pdf":
            is_png = 0
        if filename[-4:] in ['.pdf', '.png']:
            filename = filename[:-4]
    pre = plt if fig is None else fig
    f1 = os.path.join(PLOT_DIR, filename+'.pdf')
    f2 = os.path.join(PLOT_DIR, filename+'.png')
    pre.savefig(f1, **kwargs)
    print(f1, 'saved.')
    if is_png:
        pre.savefig(f2, dpi=300 if dpi is None else dpi, **kwargs)
    print(f2, 'saved.')

save = save_pdfpng
save_plot = save_pdfpng

def text_top_center(ax, text, **kwargs):
    """ write text on the top center of the figure """

    ax.text(0.5, 0.95, text, va='top', ha='center',
            transform=ax.transAxes, **kwargs)
    # ax.annotate(text, xy=(x, y), xycoords='data',
    #             xytext=(hspace, 0), textcoords='offset points',
    #             va='center', **kwargs)
    return


def set_y_decades(decades, ax=None):
    if ax is None:
        ax = plt.gca()
    ymin = ax.get_ylim()[1] / 10**decades
    ax.set_ylim(bottom=ymin)
