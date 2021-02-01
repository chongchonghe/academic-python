# -*- coding: utf-8 -*-
"""plottools.py
Tools for plotting with matplotlib. Use along with scienceplot.
Author: Chong-Chong He

"""

import os
import __main__
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import ndimage
import json
import datetime

PLOT_DIR = '.'
SAVING = True

DASHES = [[], # solid
          [3,3], # dash
          [1,1], # dot
          [1,1,3,1], # dot dash
          # [9,3], # long dash
          [6,2], # long dash
          [3,1,1,1,1,1], # dash dot dot
          [9,3,3,3], # long short dash
          [1,1,9,1], # dot long dash
          [9,3,3,3,3,3], # long short short dash
          [9,1,1,1,1,1], # long dash dot dot
          [3,3,3,3,1,3], # dash dash dot
          [9,3,9,3,1,3], # long dash long dash dot
          [9,3,9,3,3,3], # long dash long dash dash
          [9,3,3,3,1,3], # long dash dash dot
          [9,1,1,1,1,1,1,1], # long dash dot dot dot
         ]


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

def clean_sharex(axes=None, hs=0.02):
    """
    For figure with multiple axes that share the x-axis, only
    keep the tick labels for the lower one, and remove the vertical
    space between the axes.
    """
    if axes is None:
        axes = plt.gcf().axes
    plt.subplots_adjust(hspace=hs)
    if axes.ndim == 1:
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_xlabel('')
    elif axes.ndim == 2:
        for axrow in axes[:-1]:
            for ax in axrow:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_xlabel('')

def clean_sharey(axes=None, keep_tick_label=False, ws=0.02):
    """
    For figure with multiple axes that share the y-axis, only
    keep the tick labels for the left one, and remove the horizontal
    space between the axes.
    """
    if axes is None:
        axes = plt.gcf().axes
    plt.subplots_adjust(wspace=ws)
    if np.ndim(axes) == 1:
        for ax in axes[1:]:
            if not keep_tick_label:
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel('')
    # elif axes.ndim == 2:
    elif np.ndim(axes) == 2:
        for axcol in axes:
            for ax in axcol[1:]:
                if not keep_tick_label:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_ylabel('')

def shared_ylabel(axes, text=None):
    """
    For a figure with multiple axes that share the x-axis *and*
    have the same ylabel, place one ylabel at the center with
    given text.
    """
    # if axes is None:
    #     axes = gcf().axes
    if np.ndim(axes) == 2:
        axes = [ax[0] for ax in axes]
    if text is None:
        text = axes[0].get_ylabel()
    for _ax in axes:
        _ax.set_ylabel('')
    size = len(axes)
    yl = axes[int(size/2)].set_ylabel(text)

    if size%2 == 0: # Even number of axes, move label to center
        yl.set_position((yl.get_position()[0],1))
        # yl.set_verticalalignment('bottom')
        yl.set_horizontalalignment('center')

def scaled_figure(rows=1, columns=1, **kwargs):
    f, ax = plt.subplots(rows, columns, **kwargs)
    w, h = f.get_size_inches()
    f.set_size_inches(w * columns, h * rows)
    return f, ax

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

# def save_pdfpng(filename, fig=None, dpi=None, isprint=1, **kwargs):
def save_pdfpng(filename, fig=None, dpi=None, isprint=1, fromfile=None, **kwargs):
    """
    Save the current figure to PDF and PNG in PLOT_DIR.
    PLOT_DIR and TAG are used

    If filename has extension '.pdf', only a PDF will be saved. Otherwise
    save both PDF and PNG.
    """

    if not SAVING:
        return
    if dpi is None:
        dpi = 300
    with_ext = False
    if len(filename) > 4:
        if filename[-4:] in ['.pdf', '.png']:
            with_ext = True
            ext = filename[-4:]
    if with_ext:
        basename = filename[:-4]
    else:
        basename = filename
    pre = plt if fig is None else fig
    if with_ext:
        if ext == '.png':
            fn = os.path.join(PLOT_DIR, filename)
            pre.savefig(fn, dpi=dpi, **kwargs)
            if isprint:
                print(fn, 'saved.')
        else:
            # PDF, no dpi
            fn = os.path.join(PLOT_DIR, filename)
            pre.savefig(fn, **kwargs)
            if isprint:
                print(fn, 'saved.')
    else:
        # os.makedirs(os.path.join(PLOT_DIR, 'pngs'), exist_ok=1)
        os.makedirs(os.path.join(PLOT_DIR, 'pdfs'), exist_ok=1)
        f1 = os.path.join(PLOT_DIR, 'pdfs', filename+'.pdf')
        f2 = os.path.join(PLOT_DIR, filename+'.png')
        pre.savefig(f1, **kwargs)
        pre.savefig(f2, dpi=dpi, **kwargs)
        if isprint:
            print(f1, 'saved.')
            print(f2, 'saved.')

    # write plotting logs into info.json
    fn_json = os.path.join(PLOT_DIR, "info.json")
    if not os.path.isfile(fn_json):
        data = {"_About": ("This is a log file that tells what scripts produce "
                           "the figures in this folder. TODO: enable relative "
                           "path to the 'from' file.")}
    else:
        with open(fn_json, 'r') as ff:
            data = json.load(ff)
    thisdic = {"data": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               # "from": fromfile if fromfile is not None else "Unspecified"}
               "figure path": PLOT_DIR,
               "from": str(os.path.basename(__main__.__file__))}
    data[basename] = thisdic
    with open(fn_json, 'w') as ff:
        json.dump(data, ff, indent=2, sort_keys=True)

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


def set_y_decades(decades, ax=None, is_ret_ymax=0):
    if ax is None:
        ax = plt.gca()
    # a very lazy way, sufficient when decades is small
    #ymin = ax.get_ylim()[1] / 10**decades
    #ax.set_ylim(bottom=ymin)
    themax = -1. * float('inf')
    for line in ax.lines:
        themax = max(themax, max(line.get_ydata()))
    margin = decades / 12
    ax.set_ylim(themax * 10**(margin - decades), themax * 10**margin)
    if is_ret_ymax:
        return themax

def set_y_height(height, ax=None, is_ret_ymax=0):
    if ax is None:
        ax = plt.gca()
    # a very lazy way, sufficient when decades is small
    #ymin = ax.get_ylim()[1] / 10**decades
    #ax.set_ylim(bottom=ymin)
    themax = -1. * float('inf')
    for line in ax.lines:
        themax = max(themax, max(line.get_ydata()))
    margin = height / 12
    ax.set_ylim(themax + margin - height, themax + margin)
    if is_ret_ymax:
        return themax

def myLogFormat(y, pos):
    """Usage:
    >>> import matplotlib.ticker as ticker
    >>> ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    """

    # Find the number of decimal places required
    decimalplaces = int(np.maximum(-np.log10(y),0))     # =0 for numbers >=1
    # Insert that number into a format string
    formatstring = '{{:.{:1d}f}}'.format(decimalplaces)
    # Return the formatted tick label
    return formatstring.format(y)

def setMyLogFormat(ax, axis='y'):
    """ Usage:
    >>> setMyLogFormat(ax, 'y')
    >>> setMyLogFormat(ax, 'both')
    """

    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(myLogFormat))

def plot_ind_cbar(ax=None, fn="colorbar.pdf", cmap='viridis', lims=[0, 1], label=''):
    """ Plot a individual colorbar, for demonstration of the color scale """

    import plotutils as pu
    if ax is None or ax == 'new':
        f, axhid = sized_figure(figsize=[4, 1])
        axhid.axis('off')
        ax = f.add_axes([.1, .5, .8, .4])
    ax0 = f.add_axes([-1, -1, .1, .1])
    im = ax0.imshow([lims], cmap=cmap)
    cb = plt.colorbar(im, cax=ax, orientation='horizontal')
    cb.set_label(label)
    plt.savefig(fn)
    return

def plot_ind_cbar(ax=None, cmap='viridis', lims=[0, 1], label='',
                  is_use_plotutils=False):
    """ Plot a individual colorbar, for demonstration of the color scale """

    if_new = False
    if ax is None or ax == 'new':
        is_new = True
        if is_use_plotutils:
            import plotutils as pu
            f, axhid = sized_figure(figsize=[4, 1])
            axhid.axis('off')
        else:
            f = plt.figure(figsize=[5, 1])
        ax = f.add_axes([.1, .5, .8, .4])
    ax0 = f.add_axes([.5, .5, .1, .1])
    im = ax0.imshow([lims], cmap=cmap)
    ax0.set_visible(0)
    cb = plt.colorbar(im, cax=ax, orientation='horizontal')
    cb.set_label(label)
    if is_new:
        return f

def my_legend(axis = None):

    if axis == None:
        axis = plt.gca()

    N = 32
    Nlines = len(axis.lines)
    #print(Nlines)

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    # the 'point of presence' matrix
    pop = np.zeros((Nlines, N, N), dtype=np.float)

    for l in range(Nlines):
        # get xy data and scale it to the NxN squares
        xy = axis.lines[l].get_xydata()
        xy = (xy - [xmin,ymin]) / ([xmax-xmin, ymax-ymin]) * N
        xy = xy.astype(np.int32)
        # mask stuff outside plot
        mask = (xy[:,0] >= 0) & (xy[:,0] < N) & (xy[:,1] >= 0) & (xy[:,1] < N)
        xy = xy[mask]
        # add to pop
        for p in xy:
            pop[l][tuple(p)] = 1.0

    # find whitespace, nice place for labels
    ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0
    # don't use the borders
    ws[:,0]   = 0
    ws[:,N-1] = 0
    ws[0,:]   = 0
    ws[N-1,:] = 0

    # blur the pop's
    for l in range(Nlines):
        pop[l] = ndimage.gaussian_filter(pop[l], sigma=N/5)

    for l in range(Nlines):
        # positive weights for current line, negative weight for others....
        w = -0.3 * np.ones(Nlines, dtype=np.float)
        w[l] = 0.5

        # calculate a field
        p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
        plt.figure()
        plt.imshow(p, interpolation='nearest')
        plt.title(axis.lines[l].get_label())

        pos = np.argmax(p)  # note, argmax flattens the array first
        best_x, best_y =  (pos / N, pos % N)
        x = xmin + (xmax-xmin) * best_x / N
        y = ymin + (ymax-ymin) * best_y / N


        axis.text(x, y, axis.lines[l].get_label(),
                  horizontalalignment='center',
                  verticalalignment='center')
