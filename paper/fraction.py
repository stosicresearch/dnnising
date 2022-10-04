import numpy as np
import plots
import glob
import os
import spins

models = ['opt-125m','gpt2','bert-base','vit-base']

x = {}
y = {}

for m in models:
    files = glob.glob('fraction/'+m+'/*.dat')
    fractions = []
    widths = []
    for file in files:
        dos = np.loadtxt(file)[:,0]
        f = float(file.split('fraction')[-1].split('.dat')[0])

        dosmin = np.min(dos)
        dosmax = np.max(dos)
        w = np.abs(dosmax - dosmin)
        fractions.append(f)
        widths.append(w)
        print(m,f,w)

    #widths = (widths - np.min(widths)) / (np.max(widths) - np.min(widths))

    x[m] = np.asarray(fractions)
    y[m] = np.asarray(widths)
    ind = np.argsort(x[m])
    x[m] = x[m][ind]
    y[m] = y[m][ind] / spins.n[m]

plots.plot_single(x, y, 'fraction', xlabel='$f$', ylabel='$W$', xmin=0, xmax=1, marker=['o-','s-','D-','^-'], colors=['tab:green','tab:orange','tab:purple','tab:cyan'], grid=True, axissize=12, legendloc='upper right', extension='.png')#, fontsize=32)    
