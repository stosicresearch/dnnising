import numpy as np
import plots
import os
import spins

#model = 'opt-125m'
#model = 'opt-350m'
#model = 'opt-1.3b'
#model = 'bert-base'
#model = 'bert-large'
#model = 'beit-base'
#model = 'deit-base'
#model = 'vit-base'
#model = 'gpt2'
#model = 'gpt2-medium'
model = 'bloom'
names = [model,model+'_shuffled']

x = {}
y = {}

for n in names:
    data = np.loadtxt('dos/'+n+'.dat')
    print(n,data.shape)

    x[n] = data[:,0] / spins.n[model]
    y[n] = data[:,1]

plots.plot_single(x, y, 'dos_'+model, xlabel='$E/N$', ylabel='$\ln\, g(E)\, /\, N$', marker=['o-','^-'], colors=['black', 'tab:red'], grid=True, fontsize=12, bbox_inches='tight', extension='.png', legendloc='upper right')

    