import numpy as np
import plots
import glob
import os

# Critical temperature Tc
files = glob.glob('specificheat/*.dat')
for f in files:
    data = np.loadtxt(f)

    t = data[:,0]
    c = data[:,1]

    index=np.argmax(c)
    tc = t[index]

    m = f.split('\\')[1].split('.')[0].split('C_')[1]
    print(m,tc)


# Plot C(t) vs T.
x = {}
y = {}

#models = ['gpt2']
#models = ['gpt2-medium']
#models = ['bloom']
#models = ['opt-125m']
#models = ['opt-350m']
#models = ['opt-1.3b']
#models = ['bert-base']
#models = ['bert-large']
#models = ['vit-base']
#models = ['deit-base']
models = ['beit-base']
models.append(models[0]+'_shuffled')
for m in models:
    y[m] = []

    data = np.loadtxt('specificheat/C_'+m+'.dat')

    t = data[:,0]
    c = data[:,1]

    x[m] = t
    y[m] = c * 100

plots.plot_single(x, y, 'C_'+models[0], xlabel='$T$', ylabel='$C(T)\;\; (J\,/\,T\; 10^{-2})$', marker=['o-','^-','s-','P-'], colors=['tab:cyan', 'tab:green', 'tab:pink', 'tab:brown'], grid=True, axissize=12, markersize=3, legendloc='upper right', xmin=0, xmax=0.4, ymin=1.1, ymax=1.8, extension='.png')#, bbox_inches="tight")

    