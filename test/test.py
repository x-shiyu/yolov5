from pathlib import Path
import os
from tqdm import tqdm
import logging
import time
import torch
import subprocess
import platform
from torch import optim
from contextlib import contextmanager
from multiprocessing import Process
import torch.distributed as dist
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

# def check_git_status():
#     # Suggest 'git pull' if repo is out of date
#     if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
#         s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
#         if 'Your branch is behind' in s:
#             print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')
#
# check_git_status()
@contextmanager
def foo(rank):
    if rank != 0:
        torch.distributed.barrier()
    print("starting...")
    yield 4
    if rank == 0:
        torch.distributed.barrier()
        print("res")





def run(rank, size):
    pass


def init_processes(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29555'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.manual_seed(1)
    fn(rank, size)
    print("MM")
    print(dist.get_rank())
    print(dist.get_world_size())
    print(dist.is_available())


def main():
    size = 2
    processes=[]
    for i in range(size):
        p = Process(target=init_processes, args=(i, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    # start_time = time.time()
    # main()
    # end_time = time.time()
    # print("耗时：", end_time-start_time)
    import yaml
    with open('../models/yolov5l.yaml') as f:
        res = yaml.load(f, Loader=yaml.FullLoader)
        anchors = res['anchors']
        na = (len(anchors[0]) // 2)
        pass