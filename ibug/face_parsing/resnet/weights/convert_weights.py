import torch
import sys

ckpt, out_name = sys.argv[1:3]

ckpt = torch.load(ckpt, 'cpu')

new_sd = {}
for k, v in ckpt['state_dict'].items():
     if 'backbone' in k:
         new_sd[k.replace('backbone','encoder')] = v
     elif 'aux' in k:
         continue
     else:
         new_sd[k] = v

ckpt={'state_dict':new_sd}
torch.save(ckpt, out_name)