import numpy as np
import matplotlib as mpl
mpl.use('Agg')
mpl.use('pdf')
import matplotlib.pylab as plt
from matplotlib.ticker import StrMethodFormatter, NullFormatter, FuncFormatter
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.0fK' % (x*1e-3)

def plot_single(x, y, name, title=None, xlabel=None, ylabel=None, marker=['-'], markersize=5, xmin=None, xmax=None, ymin=None, ymax=None, xunits='', yunits='', legendloc=None, colors=None, grid=False, fontsize=None, axissize=None, bbox_inches=None, extension='.pdf', linewidth=None, avoidlabel=None):
    plt.rc('font', family='serif')
    plt.rc('text')#, usetex=True)

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    formatter = FuncFormatter(thousands)
    fig, a = plt.subplots(1,1,figsize=(1*5,4))
    #fig, a = plt.subplots(1,1,figsize=(4,3))

    count = 0
    for k,v in y.items():
        if xunits == 'thousands':
            a.xaxis.set_major_formatter(formatter)
        if yunits == 'thousands':
            a.yaxis.set_major_formatter(formatter)
        a.tick_params(axis='x', direction='in')
        a.tick_params(axis='y', direction='in')
        a.tick_params(which='minor', axis='x', direction='in')
        a.tick_params(which='minor', axis='y', direction='in')

        label = k
        if avoidlabel and (avoidlabel in k):
            label = None

        if colors:
            a.plot(x[k], y[k], marker[count], color=colors[count], markersize=markersize, linewidth=linewidth, label=label)
        else:
            a.plot(x[k], y[k], marker[count], markersize=markersize, linewidth=linewidth, label=label)

        if xmin is not None: a.set_xlim(xmin,x[k].max())
        if (xmin is not None) and (xmax is not None): a.set_xlim(xmin,xmax)
        if (ymin is not None) and (ymax is not None): a.set_ylim(ymin,ymax)
        if axissize is not None:
            if xlabel is not None: a.set_xlabel(xlabel, fontsize=axissize)
            if ylabel is not None: a.set_ylabel(ylabel, fontsize=axissize)
        else:
            if xlabel is not None: a.set_xlabel(xlabel)
            if ylabel is not None: a.set_ylabel(ylabel)
        if legendloc is None:
            legendloc = 'upper left'
        a.legend(loc=legendloc, frameon=False)
        a.grid(grid)
        if title: fig.suptitle(title, fontsize=16)
        count += 1
    fig.savefig(name+extension, bbox_inches=bbox_inches)


def plot_overlay(x, y, name, title=None, xlabel=None, ylabel=None, marker='-', markersize=5, xmin=None, xmax=None, ymin=None, ymax=None, xunits='', yunits='', colors=None, grid=False, fontsize=None, axissize=None):
    plt.rc('font', family='serif')
    plt.rc('text')#, usetex=True)

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    formatter = FuncFormatter(thousands)
    fig, a = plt.subplots(1,1,figsize=(1*5,4))
    #fig, a = plt.subplots(1,1,figsize=(4,3))

    count = 0
    lines = ()
    labels = ()
    for k,v in y.items():
        if count > 0:
            a = a.twinx()
        if xunits == 'thousands':
            a.xaxis.set_major_formatter(formatter)
        if yunits == 'thousands':
            a.yaxis.set_major_formatter(formatter)
        a.tick_params(axis='x', direction='in')
        a.tick_params(axis='y', direction='in')
        a.tick_params(which='minor', axis='x', direction='in')
        a.tick_params(which='minor', axis='y', direction='in')


        #ax1 = plt.subplot()
        #l1, = ax1.plot(y['acc'], color='red')
        #ax2 = ax1.twinx()
        #l2, = ax2.plot(acceleration, color='orange')
        #plt.legend([l1, l2], ["speed", "acceleration"])
        #plt.show()

        if colors:
            l = a.plot(x[k], y[k], marker, color=colors[count], markersize=markersize, label=k)
        else:
            l = a.plot(x[k], y[k], marker, markersize=markersize, label=k)
        #l, la = a.get_legend_handles_labels()
        #lines += (l,)
        #labels += (k,)

        if xmin is not None: a.set_xlim(xmin,x[k].max())
        yymin = ymin[count]
        yymax = ymax[count]
        if (xmin is not None) and (xmax is not None): a.set_xlim(xmin,xmax)
        if (yymin is not None) and (yymax is not None): a.set_ylim(yymin,yymax)
        if axissize is not None:
            if xlabel is not None: a.set_xlabel(xlabel, fontsize=axissize)
            #if ylabel is not None: a.set_ylabel(ylabel[count], fontsize=axissize)
        else:
            if xlabel is not None: a.set_xlabel(xlabel)
            #if ylabel is not None: a.set_ylabel(ylabel[count])
        #a.legend(loc='upper left')
        if count == 0:
            a.grid(grid)
        if title: fig.suptitle(title, fontsize=16)
        count += 1

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.3, 0.89), frameon=False)

    #plt.legend(loc='upper left')
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(lines, labels, loc='upper center')
    fig.savefig(name+'.pdf')#, bbox_inches="tight")


def plot_multi(x, y, name, title=None, xlabel=None, ylabel=None, marker='-', markersize=5, xmin=None, xmax=None, ymin=None, ymax=None, xunits='', yunits='', legendloc=None, colors=None, grid=False, fontsize=None, axissize=None, linewidth=None):
    plt.rc('font', family='serif')
    plt.rc('text')#, usetex=True)

    if fontsize is not None:
        plt.rcParams.update({'font.size': fontsize})

    n = len(y.keys())

    formatter = FuncFormatter(thousands)
    fig, ax = plt.subplots(1,n,figsize=(1*5*n,4))
    #fig, a = plt.subplots(1,1,figsize=(4,3))

    nfig = 0
    for k,v in y.items():
      count = 0
      for l,u in v.items():
        a = ax[nfig]
        if xunits == 'thousands':
            a.xaxis.set_major_formatter(formatter)
        if yunits == 'thousands':
            a.yaxis.set_major_formatter(formatter)
        a.tick_params(axis='x', direction='in')
        a.tick_params(axis='y', direction='in')
        a.tick_params(which='minor', axis='x', direction='in')
        a.tick_params(which='minor', axis='y', direction='in')

        if 'shuffled' in l:
            label = 'shuffled'
        else:
            label = 'trained'
        if colors:
            a.plot(x[k][l], y[k][l], marker[count], color=colors[count], markersize=markersize, linewidth=linewidth, label=label)
        else:
            a.plot(x[k][l], y[k][l], marker[count], markersize=markersize, linewidth=linewidth, label=label)

        if xmin is not None: a.set_xlim(xmin,x[k][l].max())
        if (xmin is not None) and (xmax is not None): a.set_xlim(xmin,xmax)
        if (ymin is not None) and (ymax is not None): a.set_ylim(ymin,ymax)
        if axissize is not None:
            if xlabel is not None: a.set_xlabel(xlabel, fontsize=axissize)
            if nfig==0:
                if ylabel is not None: a.set_ylabel(ylabel, fontsize=axissize)
        else:
            if xlabel is not None: a.set_xlabel(xlabel)
            if nfig==0:
                if ylabel is not None: a.set_ylabel(ylabel)
        if nfig==0:
            if legendloc is None:
                legendloc = 'upper left'
            a.legend(loc=legendloc, frameon=False)
        a.grid(grid)
        if title: a.set_title(title[nfig], fontsize=16)
        count += 1
      nfig += 1
    fig.savefig(name+'.pdf', bbox_inches="tight")
