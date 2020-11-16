from pathlib import Path
import os
from tqdm import tqdm
import logging
import time
import torch
import subprocess
import platform
from torch import optim

# logging.basicConfig(level=logging.INFO)
# img_path = os.path.abspath('F:\images')
#
#
# def imgRename(dir):
#     list_dir = os.listdir(dir)
#     for idx, item in enumerate(list_dir):
#         name_item_arr = item.split('.')
#         target_name = 'b_' + str(idx) + '.' + name_item_arr[len(name_item_arr) - 1]
#         os.rename(os.path.join(dir, item), os.path.join(dir, target_name))
#         print('deal:', idx + 1)
#
#
# # imgRename(img_path)
#
# def tqdmTest():
#     for i in tqdm(range(10000)):
#         time.sleep(100)
#
#
# # tqdmTest()
#
# # logger = logging.getLogger(__name__)
# # logger.info('xx')
# from torch.nn.parameter import Parameter
#
# ckpt = torch.load('../weights/yolov5s.pt')
# # state = ckpt['model'].float().state_dict()
# model = ckpt['model']
# pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
# for k, v in model.named_parameters():
#     v.requires_grad = True
#     if '.bias' in k:
#         pg2.append(v)  # biases
#     elif '.weight' in k and '.bn' not in k:
#         pg1.append(v)  # apply weight decay
#     else:
#         pg0.append(v)
#         break
# pg3 = [Parameter(torch.tensor([1, 2, 3, 4],dtype=torch.float), requires_grad=True)]
# pg4 = [Parameter(torch.tensor([2, 3, 4, 5],dtype=torch.float), requires_grad=True)]
#
# optimizer = optim.Adam(pg0, lr=0.0001, betas=(0.937, 0.999))  # adjust beta1 to momentum
# optimizer.add_param_group({'param': pg3, 'weight_decay': 0.0001})  # add pg1 with weight_decay
# optimizer.add_param_group({'params': pg4})  # add pg2 (biases)
# pass

# params = torch.tensor([1, 2, 3, 4])
# optimizer = optim.Adam([{
#     'params':params
# }])
# optimizer.add_param_group({"params"})
# pass

def check_git_status():
    # Suggest 'git pull' if repo is out of date
    if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')

check_git_status()