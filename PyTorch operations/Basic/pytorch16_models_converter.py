# convert the .pth file of PyTorch >=1.6 to fit older versions

import torch

model_cp_read = 'ssnet_epoch7_bs2.pth'
model_cp_save = 'ssnet_epoch7_bs2_.pth'

state_dict = torch.load(model_cp_read)

torch.save(model_cp_read, model_cp_save, _use_new_zipfile_serialization=False)
