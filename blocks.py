import numpy as np
import plots
import glob
import os

blocks = [1,2,4,6,8,10,12,16,24]

x = {}
y = {}

models = ['opt-125m','bert-base','gpt2']
for m in models:
    y[m] = []

    bb = []
    for b in blocks:
        path = 'blocks/'+m+'/blocks'+str(b)
        files = glob.glob('blocks/'+m+'/*.dat')
        hasfile = False
        for f in files:
            if 'blocks'+str(b) in f:
                hasfile = True
        if not hasfile:
            continue
        bb.append(b)
        trained = np.loadtxt('blocks/'+m+'/blocks'+str(b)+'.dat')[:,0]
        shuffled = np.loadtxt('blocks/'+m+'/blocks'+str(b)+'_shuffled.dat')[:,0]

        trained_min = np.min(trained)
        trained_max = np.max(trained)
        trained_width = np.abs(trained_min - trained_max)

        shuffled_min = np.min(shuffled)
        shuffled_max = np.max(shuffled)
        shuffled_width = np.abs(shuffled_min - shuffled_max)
        delta = trained_width - shuffled_width
        y[m].append(trained_width)

        print(m,b,trained_width,shuffled_width,delta)

    x[m] = np.asarray(bb)
    x[m] = x[m] / x[m][-1]
    y[m] = np.asarray(y[m])
    ymax = np.max(y[m])
    y[m] = y[m] / ymax

plots.plot_single(x, y, 'blocks', xlabel='$l\, /\, L$', ylabel='$\Delta\, /\, \Delta_{max}$', marker=['o-','^-','s-','P-'], colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'], grid=True, axissize=12, legendloc='lower right', extension='.png')