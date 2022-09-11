import numpy as np
import plots
import glob
import os
import spins

x = {}
y = {}

files = glob.glob('accuracy/*.dat')
for file in files:
    data = np.loadtxt(file)
    f = data[:,0]
    acc = data[:,1]

    m = file.split('\\')[1].split('.')[0]

    if 'imagenet' in m:
        # absolute error
        acc = acc[0] - acc
    else:
        # relative error
       acc = (acc - acc[0]) / acc[0] * 100

    print(m,f.shape,acc.shape,acc)
    x[m] = f
    y[m] = acc

plots.plot_single(x, y, 'accuracy', xlabel='$f$', ylabel='$\mathcal{E}$', xmin=0, xmax=0.5, ymin=0, ymax=20, marker=['o-','s-','D-','^-','o-','o-'], colors=['black','tab:red','tab:green','tab:brown','tab:orange','tab:blue'], grid=True, axissize=12, legendloc='upper right', extension='.png')