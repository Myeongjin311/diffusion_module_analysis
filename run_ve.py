import torch
import random
import numpy as np
from diffusers import DiffusionPipeline

import os

import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int)
parser.add_argument('--threshold', type=int)
parser.add_argument('--scale', type=float)
parser.add_argument('--bsize', type=int, default=4)
parser.add_argument('--num_steps', type=int)
parser.add_argument('--is_save', action='store_true')

parser.add_argument('--norm_b', action='store_true')

parser.add_argument('--t_start_idx', type=int)
parser.add_argument('--duration', type=int, default=2)

parser.add_argument('--select_channels', default=None, nargs='+', type=int)
parser.add_argument('--channel_inverse', action='store_true')


parser.add_argument('--gradients', action='store_true')


parser.add_argument('--down_block_indices','-down_ids', default=None, nargs='+', type=int)
parser.add_argument('--skip_block_indices','-skip_ids', default=None, nargs='+', type=int)
parser.add_argument('--up_block_indices','-up_ids', default=None, nargs='+', type=int)
parser.add_argument('--scales', default=[1.0, 1.0, 1.0], nargs='+', type=float)

parser.add_argument('--relation_check', action='store_true')

parser.add_argument('-fmap', '--feature_map_viz', action='store_true')

parser.add_argument('--mask_pth', default=None, type=str)
parser.add_argument('--mask_dir_name', default=None, type=str)

parser.add_argument('--channels', default=None, nargs='+', type=int)
parser.add_argument('--mask_paths', default=None, nargs='+', type=str)
parser.add_argument('--segment_names', default=None, nargs='+', type=str)
parser.add_argument('--iou_threshold', type=float, default=0.7)


args = parser.parse_args()

seed = args.seed
threshold = args.threshold
scale = args.scale
bsize = args.bsize
num_steps = args.num_steps
is_save = args.is_save


t_start_idx = args.t_start_idx
duration = args.duration


select_channels = args.select_channels
channel_inverse = args.channel_inverse


gradients = args.gradients

up_block_indices = args.up_block_indices
down_block_indices = args.down_block_indices
skip_block_indices = args.skip_block_indices
scales = args.scales

relation_check = args.relation_check


feature_map_viz = args.feature_map_viz

mask_pth = args.mask_pth
mask_dir_name = args.mask_dir_name

channels = args.channels

mask_paths = args.mask_paths
segment_names = args.segment_names

iou_threshold = args.iou_threshold

def create_select_pth(dir_name, select_channels):
    count = 0
    is_exist = False
    
    test_dir = dir_name + f'_{count}'
    while os.path.exists(test_dir):
        with open(os.path.join(test_dir, 'channel_indices.txt'), 'r') as f:
            lines = f.readlines()
            lines_str_list = lines[0].split()

            lines_list = list(map(lambda x: int(x), lines_str_list))
            
            #print(lines_list)
            if select_channels == lines_list:
                is_exist = True
                break      
        
        count += 1
        test_dir = dir_name + f'_{count}'
    
    return test_dir, count, is_exist

print('down_block_indices:', down_block_indices)
print('skip_block_indices:', skip_block_indices)
print('up_block_indices:', up_block_indices)

print('is_save:', is_save)

torch.manual_seed(seed) # 0
random.seed(seed) # 0
np.random.seed(seed) # 0
model_id = "google/ncsnpp-celebahq-256"
# google/ncsnpp-celebahq-256
# google/ncsnpp-ffhq-256
# load model and scheduler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# directory name for the block scalings

block_dir = '_'.join(map(lambda x: str(x), [down_block_indices, skip_block_indices, up_block_indices]))
scale_dir = '_'.join(map(lambda x: str(x), scales))
t_dir = f't_{t_start_idx}_{t_start_idx+duration}'

bs_dir = os.path.join(block_dir, t_dir, scale_dir)

if mask_dir_name is not None:
    bs_dir = os.path.join(bs_dir, mask_dir_name)

dir_name = os.path.join(f've/seed_{seed}', bs_dir)



# create directory name
if select_channels:
    sel_channels_dir = os.path.join(dir_name, 'select_channels')
    if channel_inverse:
        channel_indices = list(range(128))
        select_channels = list(filter(lambda x: x not in select_channels, channel_indices))
        
    sel_channels_dir, count, is_exist = create_select_pth(sel_channels_dir, select_channels)
    
    #dir_name = os.path.join(sel_channels_dir, f't_{t_start_idx}/b_scale_{backbone_scale}_s_scale_{skip_scale}_t_{threshold}_s_{scale}_ns_{num_inference_steps}')
    
    #os.makedirs(dir_name, exist_ok=True)
    
    txt_pth = os.path.join(sel_channels_dir, 'channel_indices.txt')
            
    if not is_exist:
        with open(txt_pth, 'w') as f:
            f.write(' '.join(map(lambda x: str(x), select_channels)))
    
    #norm_pth = f'info/norm/new_sampling/seed_{seed}/{position}/select_channels/select_channels_{count}/t_{t_start_idx}/b_scale_{backbone_scale}_s_scale_{skip_scale}_t_{threshold}_s_{scale}_ns_{num_inference_steps}'
    
else:

    if relation_check:
        dir_name = os.path.join(dir_name, 'relation')
        
        #norm_pth = f'info/norm/new_sampling/seed_{seed}/{bs_dir}/relation/'

            
print('dir_name:', dir_name)


block_indices = [down_block_indices, skip_block_indices, up_block_indices]
sde_ve = DiffusionPipeline.from_pretrained(model_id)
sde_ve = sde_ve.to(device)
# run pipeline in inference (sample random noise and denoise)

segment_masks = None
if segment_names is not None:
    segment_masks = {}
    for s_name in segment_names:
        
        paths = mask_paths[:4]
        mask_paths = mask_paths[4:]

        masks = []
        for pth in paths:
            masks.append(torch.load(pth).to(device))

        mask = torch.stack(masks)
        mask = mask.unsqueeze(dim=1)

        print('s_name:', s_name)
        print('mask shape:', mask.shape)
        segment_masks[s_name] = mask



image = sde_ve(num_inference_steps=num_steps, batch_size=bsize, seed=seed, threshold=threshold, scale=scale, dir_name=dir_name, is_save=is_save, \
    scales=scales, t_start_idx=t_start_idx, select_channels=select_channels, channel_inverse=channel_inverse, gradients=gradients, \
                   block_indices=block_indices, relation_check=relation_check, duration=duration, feature_map_viz=feature_map_viz, \
                    mask_pth=mask_pth, channels=channels, segment_masks=segment_masks, iou_threshold=iou_threshold).images[0]


