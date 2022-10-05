import numpy as np
import dnnising

# Parameters.
model = 'opt-125m'
hidden_size = 768
transformer_layers = 12
max_steps = 1000000;
num_shuffle = 10;

# Allocate and read model.
dnnising.py_alloc(transformer_layers * 4, hidden_size)
dnnising.py_read(model)

# Shuffle weights.
#dnnising.py_shuffle(num_shuffle)

# Minimize energy.
e = dnnising.py_energy()
for i in range(max_steps):
    e = dnnising.py_mcmc_step(e)
print('Final energy:', e)

# Save spins.
dnnising.py_save()
