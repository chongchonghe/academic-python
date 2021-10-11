# -*- coding: utf-8 -*-
"""plottools.py
Tools for plotting with matplotlib. Use along with scienceplot.
Author: Chong-Chong He

# About OVERWRITE

By default, OVERWRITE is on. To turn it off, do
dt.turnoff_overwrite(). Then, you can use command line arguments to
turn it on in case you want to enable overwriting ocasionally. e.g.
>>> if len(sys.argv) >= 2:
>>>     if "-f" in sys.argv[1:]:
>>>         pt.turnon_overwrite()

"""

import os
import __main__
import math
import numpy as np
from math import atan, pi
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import ndimage
import json
import datetime
import logging
from scipy.interpolate import interp1d

PLOT_DIR = '.'
SAVING = True
OVERWRITE = True
BACKUP = True
FROMFILE = "Unspecified"
SUPPRESS_PRINT = 1 # 0: print a lot; 1: print none; 2: print filename
FMT = 'pdfpng'
# FMT:
# 'pdf': only save pdf
# 'png': save png; save pdf in pdfs/
# 'pdfpng': (not implemented yet) save pdf; save png in pngs/

plt.style.use(['science', 'no-latex'])

def init(file_):
    global FROMFILE
    FROMFILE = file_

set_fromfile = init

def backup(fi):
    """ Rename fi: appending a date and time in ISO 8601 format. e.g.
    '-bk20200101T120000' """
    filename, file_extension = os.path.splitext(fi)
    dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    fo = f"{filename}-bk{dt}{file_extension}"
    os.rename(fi, fo)

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


def sized_figure_sci(rows=1, columns=1, mergex=True, mergey=True,
                 rescale=1.0, top=False, right=False,
                 ret_size=False, figsize=None,
                 **kwargs):

    f = plt.figure()
    w_inch, h_inch = fig.get_size_inches()

    # ratio = 0.68
    # w_inch = 3.6 * rescale
    # h_inch = w_inch * ratio
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

def turnoff_overwrite():
    global OVERWRITE
    OVERWRITE = False

def turnon_overwrite():
    global OVERWRITE
    OVERWRITE = True

def turnoff_backup():
    global BACKUP
    BACKUP = False

def turnoff_print():
    global ISPRINT
    ISPRINT = False

# def save_pdfpng(filename, fig=None, dpi=None, isprint=1, **kwargs):
def get_full_filename(filename):
    if len(filename) <= 4:
        fn = os.path.join(PLOT_DIR, filename+'.png')
    else:
        if filename[-4:] not in ['.pdf', '.png']:
            fn = os.path.join(PLOT_DIR, filename+'.png')
        else:
            fn = os.path.join(PLOT_DIR, filename)
    return fn

def save_pdfpng(filename, fig=None, dpi=None, isprint=1, fromfile=None,
                is_overwrite=None, **kwargs):
    """
    Save the current figure to PDF and PNG in PLOT_DIR.
    PLOT_DIR and TAG are used

    If filename has extension '.pdf', only a PDF will be saved. Otherwise
    save both PDF and PNG.
    """

    if not SAVING:
        print("SAVING is false. Skiping...")
        return
    if dpi is None:
        dpi = 300
    isprint = isprint and not SUPPRESS_PRINT
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

    # write plotting logs into info.json
    fn_json = os.path.join(PLOT_DIR, "info.json")
    jsonexist = False
    if os.path.isfile(fn_json):
        if os.stat(fn_json).st_size > 0:
            jsonexist = True
            with open(fn_json, 'r') as ff:
                data = json.load(ff)
    if with_ext:
        f2 = os.path.join(PLOT_DIR, filename)
        if ext == '.png':       # set dpi
            pre.savefig(f2, dpi=dpi, **kwargs)
        else:                   # PDF, no dpi
            pre.savefig(f2, **kwargs)
        if isprint:
            print(f2, 'saved.')
    else:
        if FMT == 'png':
            # os.makedirs(os.path.join(PLOT_DIR, 'pngs'), exist_ok=1)
            os.makedirs(os.path.join(PLOT_DIR, 'pdfs'), exist_ok=1)
            f1 = os.path.join(PLOT_DIR, 'pdfs', filename+'.pdf')
            pre.savefig(f1, **kwargs)
            f2 = os.path.join(PLOT_DIR, filename+'.png')
            if not OVERWRITE and os.path.exists(f2):
                return
            backup_dir = "history_versions"
            if BACKUP and os.path.exists(f2): # make a backup
                os.makedirs(os.path.join(PLOT_DIR, backup_dir), exist_ok=1)
                # read create time from info.json
                creation = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                if jsonexist:
                    if basename in data.keys():
                        creation = data[basename]["creation time"]
                        creation = creation.replace(' ', '')
                os.rename(f2, f"{PLOT_DIR}/{backup_dir}/{filename}.{creation}.png")
            pre.savefig(f2, dpi=dpi, **kwargs)
            if isprint:
                print(f1, 'saved.')
                print(f2, 'saved.')
        elif FMT == 'pdf':
            f2 = os.path.join(PLOT_DIR, filename+'.pdf')
            if not OVERWRITE and os.path.exists(f2):
                return
            backup_dir = "history_versions"
            if BACKUP and os.path.exists(f2): # make a backup
                os.makedirs(os.path.join(PLOT_DIR, backup_dir), exist_ok=1)
                # read create time from info.json
                creation = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                if jsonexist:
                    if basename in data.keys():
                        creation = data[basename]["creation time"]
                        creation = creation.replace(' ', '')
                os.rename(f2, f"{PLOT_DIR}/{backup_dir}/{filename}.{creation}.pdf")
            pre.savefig(f2, **kwargs)
            if isprint:
                print(f2, 'saved.')
        elif FMT == 'pdfpng':
            os.makedirs(os.path.join(PLOT_DIR, 'pngs'), exist_ok=1)
            f1 = os.path.join(PLOT_DIR, filename+'.pdf')
            if not OVERWRITE and os.path.exists(f1):
                return
            pre.savefig(f1, **kwargs)
            f2 = os.path.join(PLOT_DIR, 'pngs', filename+'.png')
            pre.savefig(f2, dpi=dpi, **kwargs)
            if isprint:
                print(f1, 'saved.')
                print(f2, 'saved.')

    if not jsonexist:
        data = {"_About": ("This is a log file that tells what scripts produce "
                           "the figures in this folder. TODO: enable relative "
                           "path to the 'from' file.")}
    if fromfile is None:
        fromfile = FROMFILE
    thisdic = {"creation time": datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
               # "from": fromfile if fromfile is not None else "Unspecified"}
               "figure path": PLOT_DIR,
               "from": fromfile}
    # data[f1] = thisdic
    data[basename] = thisdic
    with open(fn_json, 'w') as ff:
        json.dump(data, ff, indent=2, sort_keys=True)
    if SUPPRESS_PRINT == 2:
        print(f2)
    return f2

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
         ]          # Author: Laurens Keek

class ToSave:
    """
    Usage:

    s = ToSave("filename")
    if s.plot:
        f = plt.figure()
        plt.plot(...)
        ...
        s.save()
    """

    def __init__(self, fn, is_always_save=False):
        self.fn = fn
        full_fn = get_full_filename(fn)
        this_overwrite = OVERWRITE if not is_always_save else True
        self.plot = not os.path.exists(full_fn)
        if this_overwrite:
            self.plot = True

    def save(self, fn=None, **kwargs):
        if fn is None:
            save_pdfpng(self.fn, **kwargs)
        else:
            save_pdfpng(fn, **kwargs)
        return

def to_latex(f, n=2):
    assert type(n) == int
    fmt = "{{:.{}g}}".format(n)
    float_str = fmt.format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if base == '1':
            return r"10^{{{}}}".format(int(exponent))
        else:
            return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

def changing_color_curve(ax, x, y, xlim, colors, lw='regular'):
    """
    Args:
        x (list):
        y (list):
        xlim (list with length 2): set the limit of x with respect to 0
            and 1 of the color scael
        colors (function): returns the color given a value from 0 to 1

    """
    interp = interp1d(x, y, kind='linear')
    x_new = np.linspace(np.min(x), np.max(x), 1000)
    y_new = interp(x_new)
    lw = {'regular': 2, 'thin': 1, 'thick': 4}[lw]
    values = (x_new - xlim[0]) / (xlim[1] - xlim[0])
    ax.scatter(x_new, y_new, c=colors(values), edgecolor='none', s=lw)

def mylegend(ax=None, loc=0.3, kind='top', align=True, nslope=1,
             margin_right=0.05, **kwargs):
    """
    Args:
        loc (float or tuple): location (x axis) of the labels. If tuple,
            ranging from loc[0] to loc[1].

    """
    if isinstance(loc, float):
        x_start = loc
        x_end = loc
    else:
        x_start = loc[0]
        x_end = loc[1]
    if kind == 'top':
        trans = ax.get_yaxis_transform()
        lines = ax.lines
        #if pos is None:
            #pos = 1 - 0.618
        poss = [x_start + i / (len(lines) - 1) * (x_end - x_start) for i in range(len(lines))]
        for pos, line in zip(poss, lines):
            xy = line.get_xydata()
            xdata = xy[:, 0]
            ydata = xy[:, 1]
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            datax = xmin + pos * (xmax - xmin)
            assert datax > -1e10
            pick = np.argmax(xy[:, 0] >= datax)
            ypick = xy[pick, 1]
            if ypick > ymax or ypick < ymin:
                continue
            assert not np.isnan(ypick), f'datax = {datax}, pos={pos}, lim={xmin}, {xmax}, '

            # align text
            if align:
                #Compute the slope
                half = nslope // 2
                dx = xdata[pick + nslope - half] - xdata[pick - half]
                dy = ydata[pick + nslope - half] - ydata[pick - half]
                if dx == 0.0:
                    ang = 0.0
                else:
                    ang = atan(dy/dx) * 180 / pi

                # Transform to screen co-ordinates
                pt = np.array([xdata[pick], ydata[pick]]).reshape((1,2))
                trans_angle = ax.transData.transform_angles(
                    np.array((ang,)), pt)[0]

            else:
                trans_angle = 0

            ax.annotate(line.get_label(), xy=(pos, ypick), xycoords=trans,
                        xytext=(0, 2), textcoords='offset points',
                        va='bottom', ha='center',
                        color=line.get_color(),
                        rotation=trans_angle,
                        **kwargs,
                       )
            # ax.text(pos, ypick, line.get_label(),
            #         transform=trans,
            #         va='bottom', ha='center',
            #         color=line.get_color(),
            #         rotation=trans_angle,
            #         **kwargs,
            # )
    elif kind == 'last':
        lines = ax.lines
        for line in lines:
            xy = line.get_xydata()
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            if xy[-1, 1] > ymax or xy[-1, 1] < ymin:
                continue
            ax.text(xy[-1, 0], xy[-1, 1], ' ' + line.get_label(), va='center',
                    color=line.get_color(),
                    **kwargs,)
        # increase xlim by 10%
        ax.set_xlim(xmin, xmax + margin_right * (xmax - xmin))

class CurvedText(matplotlib.text.Text):
    """
    A text object that follows an arbitrary curve.

    https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib
    """
    def __init__(self, x, y, text, axes, **kwargs):
        if '$' in text:
            print("Warning: CurvedText does not support LaTeX.")
            return
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = matplotlib.text.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = matplotlib.text.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used
