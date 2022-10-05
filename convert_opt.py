import torch
import transformers
import numpy as np
import os

model_name = "opt-125m"
layers = 12
#model_name = "opt-350m"
#layers = 24
#model_name = "opt-1.3b"
#layers = 24
#model_name = "opt-2.7b"
#layers = 32

outdir = model_name
os.mkdir(outdir)

model = transformers.OPTModel.from_pretrained("facebook/" + model_name)

print(model)

for i in range(layers):
    print(i)
    # Get weights.
    q = model.decoder.layers[i].self_attn.q_proj.weight.float().t().contiguous()
    k = model.decoder.layers[i].self_attn.k_proj.weight.float().t().contiguous()
    v = model.decoder.layers[i].self_attn.v_proj.weight.float().t().contiguous()
    proj = model.decoder.layers[i].self_attn.out_proj.weight.float().t().contiguous()
    fc1 = model.decoder.layers[i].fc1.weight.float().t().contiguous()
    fc2 = model.decoder.layers[i].fc2.weight.float().t().contiguous()

    # Add query, keys, and values.
    qkv = q + k + v

    # Convert to numpy.
    qkv = qkv.detach().cpu().numpy()
    proj = proj.detach().cpu().numpy()
    fc1 = fc1.detach().cpu().numpy()
    fc2 = fc2.detach().cpu().numpy()

    # Save to txt.
    np.savetxt(outdir + 'layers.' + str(i) + '.qkv' + '.txt', qkv)
    np.savetxt(outdir + 'layers.' + str(i) + '.proj' + '.txt', proj)
    np.savetxt(outdir + 'layers.' + str(i) + '.fc1' + '.txt', fc1)
    np.savetxt(outdir + 'layers.' + str(i) + '.fc2' + '.txt', fc2)