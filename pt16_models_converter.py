import torch

model_cp = 'ssnet_epoch7_bs2.pth'
model_cp2 = 'ssnet_epoch7_bs2_.pth'

state_dict = torch.load(model_cp)

torch.save(state_dict, model_cp2, _use_new_zipfile_serialization=False)
