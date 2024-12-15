# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...utils import BaseOutput
from ..embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

import pickle
import os

from math import floor

import matplotlib.pyplot as plt

import numpy as np
from scipy import ndimage

import torch.nn.functional as F

@dataclass
class UNet2DOutput(BaseOutput):
    """
    The output of [`UNet2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor


class UNet2DModel(ModelMixin, ConfigMixin):
    r"""
    A 2D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`):
            Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`):
            Tuple of downsample block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`):
            Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        downsample_type (`str`, *optional*, defaults to `conv`):
            The downsample type for downsampling layers. Choose between "conv" and "resnet"
        upsample_type (`str`, *optional*, defaults to `conv`):
            The upsample type for upsampling layers. Choose between "conv" and "resnet"
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for normalization.
        attn_norm_num_groups (`int`, *optional*, defaults to `None`):
            If set to an integer, a group norm layer will be created in the mid block's [`Attention`] layer with the
            given number of groups. If left as `None`, the group norm layer will only be created if
            `resnet_time_scale_shift` is set to `default`, and if created will have `norm_num_groups` groups.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim` when performing class
            conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        #f = open('info/downblocks.txt', 'w')
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)
            #f.write(f'down_block_type: {down_block_type}, layers_per_block + 1: {layers_per_block + 1}, \
                # input_channel: {input_channel}, output_channel: {output_channel}, \
                #     time_embed_dim: {time_embed_dim}, not is_final_block: {not is_final_block} \n\n')
            
            
        #f.close()

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        
        #f = open('info/upblocks.txt', 'w')
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
            #f.write(f'up_block_type: {up_block_type}, layers_per_block + 1: {layers_per_block + 1}, \
                # input_channel: {input_channel}, output_channel: {output_channel}, prev_output_channel: {prev_output_channel}, \
                #     time_embed_dim: {time_embed_dim}, not is_final_block: {not is_final_block} \n\n')
            
        #f.close()    

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    
    # def down_hooks(module, grad_input, grad_output):
    #     global down_in_list
    #     global down_out_list
        
    #     down_in_list.append(grad_input)
    #     down_out_list.append(grad_output)
        
        
    # def up_hooks(module, grad_input, grad_output):
    #     global up_list

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        time_idx: int = 0,
        seed: int = 0,
        threshold: int = 0,
        scale: float = 0.0,
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        scales: list = None,
        normalized_scale: bool = False,
        select_channels: list = None,
        cal_grad: bool = False,
        scheduler = None, # needed for grad
        schedule_t: float = None, # needed for grad
        num_inference_steps: int = None,
        block_indices: list = None,
        fmap_pth: str = None,
        mask: torch.Tensor = None,
        mask_pth: str = None,
        channels: list = None,
        segment_masks: dict = None,
        iou_threshold: float = 0.8,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        sample0 = sample.clone()
        
        #print('-'*8+'cal_grad:'+str(cal_grad)+'-'*8)
        

        sample.requires_grad_(True)
        #print('first sample grad:', sample.requires_grad)

        # 0. center input if necessary
        if self.config.center_input_sample:
            #print('second aaaa_sample grad:', sample.requires_grad)
            #print('grad enabled:', torch.is_grad_enabled())

            sample = 2 * sample - 1.0
            

            
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        down_in_list = []
        down_out_list = []

        mid_in_list = []
        mid_out_list = []

        up_in_list = []
        up_out_list = []
        
        def down_hooks(module, grad_input, grad_output):
            # global down_in_list
            # global down_out_list
            
            down_in_list.append(grad_input)
            down_out_list.append(grad_output[0].detach().cpu())
            
        def mid_hooks(module, grad_input, grad_output):

            mid_in_list.append(grad_input)
            
            #print('mid output:', grad_output[0])
            mid_out_list.append(grad_output[0].detach().cpu())

        def up_hooks(module, grad_input, grad_output):
            up_in_list.append(grad_input)
            up_out_list.append(grad_output[0].detach().cpu())

        def compute_fft_log_amplitude(image):
            # 이미지에 2D FFT 적용
            fft_image = torch.fft.fft2(image)
            
            # FFT 결과의 진폭 계산 및 로그 스케일 적용
            amplitude = torch.abs(fft_image)
            log_amplitude = torch.log(amplitude + 1e-8)  # 로그 계산을 위한 작은 값 추가
            
            # DC 성분을 중앙으로 이동 (fftshift)
            log_amplitude_shifted = torch.fft.fftshift(log_amplitude)
            return log_amplitude_shifted

        def fourier_save(image):
            freq = compute_fft_log_amplitude(image)
            center0 = freq.shape[0] // 2
            center1 = freq.shape[1] // 2

            freqs = []


            prev_sum = 0
            for i in range(1, center0 + 1):
                
                freq_num = freq[center0 - i: center0 + i, center1 - i: center1 + i]

                freq_sum = freq_num.sum()

                num = (2*i)**2 - (2*(i-1))**2
                freq_val = (freq_sum - prev_sum) / num

                freqs.append(freq_val.item())

                prev_sum = freq_sum

            frequency = [(2*torch.pi / freq.shape[0]) * i for i in range(1, center0+1)]


            
            return frequency, freqs

        def edge_detect(img, algorithm='prewitt'):
            if algorithm == 'prewitt':
                prewitt_h = ndimage.prewitt(img, axis=0)  # 수평 방향 edge
                prewitt_v = ndimage.prewitt(img, axis=1)  # 수직 방향 edge
                prewitt = np.sqrt(prewitt_h**2 + prewitt_v**2)  # edge magnitude 계산
            
            return prewitt
        
        def edge_diff(fmap, current=False, skip_sample=None, time_idx=None, save_pth=None):
            # load output image
            img = plt.imread(f've/seed_10/[6]/b_scale_1.00_s_scale_1.00_ns_400/image_400.png')
            img = img.sum(axis=2)

            # edge detection
            edge = edge_detect(img)

            # standardize
            s_edge = (edge - np.mean(edge)) / np.std(edge)

            if current:
                cur_img = plt.imread(f've/seed_10/[6]/b_scale_1.00_s_scale_1.00_ns_400/image_{time_idx+1}.png')
                cur_img = cur_img.sum(axis=2)

                # edge detection
                cur_edge = edge_detect(cur_img)

                # standardize
                s_cur_edge = (cur_edge - np.mean(cur_edge)) / np.std(cur_edge)

            # fmap processing
            fmap_img = fmap.detach().cpu()
            fmap_img = fmap_img.sum(dim=1)
            norm_fmap_img = (fmap_img - torch.min(fmap_img)) / (torch.max(fmap_img) - torch.min(fmap_img))
            norm_fmap_img = torch.cat([torch.cat([norm_fmap_img[0], norm_fmap_img[1]], dim=0), torch.cat([norm_fmap_img[2], norm_fmap_img[3]], dim=0)], dim=1)

            # standardize
            norm_fmap_img = norm_fmap_img.numpy()
            norm_fmap_img = 1 - norm_fmap_img
            s_fmap_img = (norm_fmap_img - np.mean(norm_fmap_img)) / np.std(norm_fmap_img)

            if save_pth:
                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                
                im0 = axs[0].imshow(s_fmap_img*2, cmap='coolwarm', vmin=-10, vmax=10)
                plt.colorbar(im0, ax=axs[0])
                axs[0].set_title('Normalized Down')
                axs[0].axis('off')

                im1 = axs[1].imshow(s_edge*2, cmap='coolwarm', vmin=-10, vmax=10)
                plt.colorbar(im1, ax=axs[1])
                axs[1].set_title('Normalized Prewitt Edge Detection')
                axs[1].axis('off')
                
                plt.tight_layout()
                os.makedirs(os.path.join(save_pth, f'edge_comparison'), exist_ok=True)
                plt.savefig(os.path.join(save_pth, f'edge_comparison/{time_idx+1}.png'))
                plt.close()

            ret_diff = np.mean(np.abs(s_edge - s_fmap_img))
            if current:
                cur_ret_diff = np.mean(np.abs(s_cur_edge - s_fmap_img))

                

            # skip processing
            if skip_sample is not None:
                skip_img = skip_sample.detach().cpu()
                skip_img = skip_img.sum(dim=1)
                norm_skip_img = (skip_img - torch.min(skip_img)) / (torch.max(skip_img) - torch.min(skip_img))
                norm_skip_img = torch.cat([torch.cat([norm_skip_img[0], norm_skip_img[1]], dim=0), torch.cat([norm_skip_img[2], norm_skip_img[3]], dim=0)], dim=1)
                
                # standardize
                norm_skip_img = norm_skip_img.numpy()
                norm_skip_img = 1 - norm_skip_img
                s_skip_img = (norm_skip_img - np.mean(norm_skip_img)) / np.std(norm_skip_img)

                ret_diff = (ret_diff, np.mean(np.abs(s_edge - s_skip_img)))
                if current:
                    cur_ret_diff = (cur_ret_diff, np.mean(np.abs(s_cur_edge - s_skip_img)))


            if current:
                return ret_diff, cur_ret_diff
            else:
                return ret_diff

        def channel_plot(fmap, save_pth, time_idx, mask_pth, channels=None):
            # channel shape: (C,)
            fmap_img = fmap.detach().cpu()
            #channel_fmap = fmap_img.reshape(fmap_img.shape[0], fmap_img.shape[1], -1)
            mask = torch.load(mask_pth)
            mask = torch.from_numpy(mask).cpu()
            mask = F.interpolate(mask[None, None, :, :], size=fmap_img.shape[-2:], mode='nearest')
            
            mask = mask[0, 0, :, :]
            channel_fmap = fmap_img[:, :, mask == 1]
            channel_fmap = channel_fmap.mean(dim=-1)
            channel_order = channel_fmap.argsort(dim=-1)
            #print('channel_order:', channel_order)

            
            #print('channel_fmap shape:', channel_fmap.shape)


            if channels is not None:

                for channel in channels:
                    channel_pth = os.path.join(save_pth, f'channels/{channel}/')
                    os.makedirs(channel_pth, exist_ok=True)

                    channel_img = fmap_img[:, channel]
                    channel_img = torch.cat([torch.cat([channel_img[0], channel_img[1]], dim=0), torch.cat([channel_img[2], channel_img[3]], dim=0)], dim=1)

                    # plt.figure(figsize=(100, 10))
                    # plt.plot(range(128), channel_fmap[i])
                    plt.imshow(channel_img, cmap='coolwarm')
                    plt.colorbar()
                    plt.tight_layout()
                    
                    plt.axis('off')
                    plt.savefig(os.path.join(channel_pth, f't_{time_idx+1}.png'))
                    plt.close()

            os.makedirs(os.path.join(save_pth, f'channel_order'), exist_ok=True)
            with open(os.path.join(save_pth, f'channel_order/t_{time_idx+1}.txt'), 'w') as f:
                for i in range(channel_order.shape[0]):
                    f.write(f'img_{i}: {channel_order[i].tolist()}\n')

        def save_fmap(fmap, save_pth, skip_sample=None, skip_save_pth=None, time_idx=None, fourier_pth=None, abs=False):
            os.makedirs(save_pth, exist_ok=True)

            if skip_sample is not None:
                os.makedirs(skip_save_pth, exist_ok=True)

            fmap_img = fmap.detach().cpu()
            #print('fmap shape:', fmap_img.shape)

            if abs:
                fmap_img = fmap_img.abs().sum(dim=1)
            else:
                fmap_img = fmap_img.sum(dim=1)

            


            if skip_sample is not None:
                skip_img = skip_sample.detach().cpu()
                skip_img = skip_img.sum(dim=1)

                #raw_skip_pth = os.path.join(skip_save_pth, 'raw')
                #os.makedirs(raw_skip_pth, exist_ok=True)

                #torch.save(skip_img, os.path.join(raw_skip_pth, f't_{time_idx+1}.pt'))
                
            # fourier trasform
            if fourier_pth:
                fig, axs = plt.subplots(2, 2)
                for i in range(fmap_img.shape[0]):
                    frequency, freqs = fourier_save(fmap_img[i])
                
                    axs[i%2, i//2].plot(frequency, freqs, label='backbone')
                    axs[i%2, i//2].set_xlabel('frequency')
                    axs[i%2, i//2].set_ylabel('log amplitude')

                    if skip_sample is not None:
                        frequency, skip_freqs = fourier_save(skip_img[i])
                
                        axs[i%2, i//2].plot(frequency, skip_freqs, label='skip')
                        axs[i%2, i//2].set_xlabel('frequency')
                        axs[i%2, i//2].set_ylabel('log amplitude')

                        axs[i%2, i//2].legend()

                    if i == 0:
                        if skip_sample is not None:
                            ret_freq = (freqs, skip_freqs)   
                        else:
                            ret_freq = freqs 


                plt.tight_layout()
                os.makedirs(os.path.join(fourier_pth, f'before'), exist_ok=True)
                plt.savefig(os.path.join(fourier_pth, f'before/t_{time_idx+1}.png'))
                plt.close()
       


            fmap_img_min = torch.amin(fmap_img, dim=(1,2), keepdim=True) 
            fmap_img_max = torch.amax(fmap_img, dim=(1,2), keepdim=True)
            
            norm_fmap_img = (fmap_img - fmap_img_min) / (fmap_img_max - fmap_img_min + 1e-5)

            if skip_sample is not None:
                skip_img_min = torch.amin(skip_img, dim=(1,2), keepdim=True) 
                skip_img_max = torch.amax(skip_img, dim=(1,2), keepdim=True)
                norm_skip_img = (skip_img - skip_img_min) / (skip_img_max - skip_img_min + 1e-5)
            
            if fourier_pth:
                fig, axs = plt.subplots(2, 2)
                for i in range(fmap_img.shape[0]):
                    frequency, freqs = fourier_save(norm_fmap_img[i])
                
                    axs[i%2, i//2].plot(frequency, freqs)
                    axs[i%2, i//2].set_xlabel('frequency')
                    axs[i%2, i//2].set_ylabel('log amplitude')

                    if skip_sample is not None:
                        frequency, freqs = fourier_save(norm_skip_img[i])
                
                        axs[i%2, i//2].plot(frequency, freqs, label='skip')
                        axs[i%2, i//2].set_xlabel('frequency')
                        axs[i%2, i//2].set_ylabel('log amplitude')

                        axs[i%2, i//2].legend()

                plt.tight_layout()
                os.makedirs(os.path.join(fourier_pth, f'after'), exist_ok=True)
                plt.savefig(os.path.join(fourier_pth, f'after/t_{time_idx+1}.png'))
                plt.close()
            
            
            norm_fmap_img = torch.cat([torch.cat([norm_fmap_img[0], norm_fmap_img[1]], dim=0), torch.cat([norm_fmap_img[2], norm_fmap_img[3]], dim=0)], dim=1)
            
            

            plt.imshow(norm_fmap_img, cmap='coolwarm')
            plt.colorbar()
            plt.tight_layout()
            plt.axis('off')
            
            plt.savefig(os.path.join(save_pth, f't_{time_idx+1}.png'))
            plt.close()

            

            if skip_sample is not None:
                norm_skip_img = torch.cat([torch.cat([norm_skip_img[0], norm_skip_img[1]], dim=0), torch.cat([norm_skip_img[2], norm_skip_img[3]], dim=0)], dim=1)

                plt.imshow(norm_skip_img, cmap='coolwarm')
                plt.colorbar()
                plt.tight_layout()
                plt.axis('off')
                
                plt.savefig(os.path.join(skip_save_pth, f't_{time_idx+1}.png'))
                plt.close()
            
            if fourier_pth is not None:
                return ret_freq
        



        def cal_ious(fmap : torch.Tensor, masks : dict, save_pth, time_idx, threshold_r=0.7):
            mask_list = []

            for i in range(fmap.shape[0]):
                threshold = torch.quantile(fmap[i].flatten(1,2), threshold_r)
                mask_list.append((fmap[i] >= threshold).float())

            fmap_mask = torch.stack(mask_list, dim=0)
            ret = {}
                
            for key, mask in masks.items():
                # mask shape: (B, 1, H, W)
                iou_pth = os.path.join(save_pth, f'iou_{threshold_r}/{key}/')
                os.makedirs(iou_pth, exist_ok=True)
                mask = F.interpolate(mask, size=fmap.shape[-2:], mode='nearest')
                

                iou = (fmap_mask * mask).sum(dim=(2,3)) / (fmap_mask.sum(dim=(2,3)) + mask.sum(dim=(2,3)) - (fmap_mask * mask).sum(dim=(2,3)))
                ret[key] = iou

                iou = iou.cpu()

                

                fig, axs = plt.subplots(2, 2)
                for row in range(2):
                    for col in range(2):
                        idx = col * 2 + row
                        axs[row, col].bar(range(iou.shape[1]), iou[idx])
                        

                plt.tight_layout()
                plt.savefig(os.path.join(iou_pth, f't_{time_idx+1}.png'))
                plt.close()

                ## top iou fmap visualization
                top_k_values, top_k_channels = torch.topk(iou, k=5, dim=1)
                for idx in range(fmap.shape[0]):
                    for k in range(5):
                        channel_idx = top_k_channels[idx, k]
                        iou_value = top_k_values[idx, k]

                        channel_pth = os.path.join(iou_pth, f'idx_{idx}_top5/t_{time_idx+1}/')
                        os.makedirs(channel_pth, exist_ok=True)

                        channel_img = fmap[idx, channel_idx]
                        plt.imshow(channel_img.cpu(), cmap='coolwarm')
                        plt.colorbar()
                        plt.tight_layout()
                        plt.axis('off')
                        
                        plt.savefig(os.path.join(channel_pth, f'channel_{channel_idx}_iou_{iou_value:.3f}.png'))
                        plt.close()

            return ret

        # unet forward pass

        down_indices,  skip_indices, up_indices = None, None, None
        if block_indices:
            down_indices, skip_indices, up_indices = block_indices

        down_scale, skip_scale, up_scale = 1.0, 1.0, 1.0
        if scales:
            down_scale, skip_scale, up_scale = scales

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)


        f_info = dict()
        f_info['low'] = dict()
        f_info['high'] = dict()

        f_info['edge_diff'] = dict()
        f_info['current_edge_diff'] = dict()

        # 3. down
        down_block_res_samples = (sample,)
        for idx, downsample_block in enumerate(self.down_blocks):

            if cal_grad:
                d_hook = downsample_block.register_full_backward_hook(down_hooks)
            
            # down block scaling
            if down_indices:
                if idx in down_indices:
                    if mask is not None:
                        mask_ = F.interpolate(mask[None, None, :, :], size=sample.shape[-2:], mode='nearest')
                        sample[mask_.expand_as(sample) == 1] *= down_scale
                    else:
                        sample *= down_scale

            f_info['low'][f'depth_{idx}'] = dict() 
            f_info['high'][f'depth_{idx}'] = dict()

            ### visualize feature maps
            if fmap_pth:
                save_pth = os.path.join(fmap_pth, f'down/block_{idx}/')
                
                fourier_pth = os.path.join(fmap_pth, f'down/block_{idx}/fourier/')
                

                ret_freq = save_fmap(sample, save_pth, time_idx=time_idx, fourier_pth=fourier_pth)  
                
                if segment_masks is not None:
                    cal_ious(sample, segment_masks, save_pth=save_pth, time_idx=time_idx, threshold_r=iou_threshold)
                
                # edge difference
                if idx == 0:
                    ret_diff, cur_ret_diff = edge_diff(sample, current=True, time_idx=time_idx, save_pth=save_pth)
                    f_info['edge_diff']['down'] = ret_diff
                    f_info['current_edge_diff']['down'] = cur_ret_diff

                if mask_pth is not None and channels is not None:
                    channel_plot(sample, save_pth, time_idx, mask_pth, channels)

                # if idx == 1:
                #     ret_diff, cur_ret_diff = edge_diff(sample, current=True, time_idx=time_idx, save_pth=save_pth)

            else:
                _, ret_freq = fourier_save(sample.sum(dim=1)[0])

                # edge difference
                # if idx == 0:
                #     ret_diff = edge_diff(sample, time_idx=time_idx)
                #     f_info['edge_diff']['down'] = ret_diff

            


            low_len = int(0.8 * len(ret_freq))
            low_freq = np.mean(ret_freq[:low_len]) 
            high_freq = np.mean(ret_freq[low_len:])
            
            f_info['low'][f'depth_{idx}']['down'] = low_freq
            f_info['high'][f'depth_{idx}']['down'] = high_freq

            
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                         
            down_block_res_samples += res_samples


        if fmap_pth:
            save_pth = os.path.join(fmap_pth, f'down/block_7/')
            fourier_pth = os.path.join(fmap_pth, f'down/block_7/fourier/')
            
            save_fmap(sample, save_pth, time_idx=time_idx, fourier_pth=fourier_pth)

            
        # 4. mid
        if cal_grad:
            m_hook = self.mid_block.register_full_backward_hook(mid_hooks)

        sample = self.mid_block(sample, emb)

        

        down_block_res_samples = list(down_block_res_samples)
        
        # 5. up
        skip_sample = None

        s_list = []
        b_list = []
        f_list = []
        
        #norm_pth = f'info/norm/seed_{seed}_t_{threshold}_s_{scale}'
        for idx, upsample_block in enumerate(self.up_blocks):
              
            if cal_grad:
                u_hook = upsample_block.register_full_backward_hook(up_hooks)
            
            
            if up_indices:
                if idx in up_indices:
                    if mask is not None:
                        mask_ = F.interpolate(mask[None, None, :, :], size=sample.shape[-2:], mode='nearest')
                        sample[mask_.expand_as(sample) == 1] *= up_scale
                    else:
                        sample *= up_scale


            # B = sample.shape[0]
            # hidden_mean = sample.mean(1).unsqueeze(1)
            # hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True) 
            # hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
            # hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
            
            # #print('backbone_scale:', backbone_scale)
            # b_scaling = (backbone_scale - 1) * hidden_mean + 1
    

            
            
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            
            if skip_indices:
                if idx in skip_indices:
                    for i in range(len(res_samples)):
                        if mask is not None:
                            mask_ = F.interpolate(mask[None, None, :, :], size=res_samples[i].shape[-2:], mode='nearest')
                            res_samples[i][mask_.expand_as(res_samples[i]) == 1] *= skip_scale
                        else:
                            res_samples[i] *= skip_scale
            

            ### visualize feature maps
            if fmap_pth:
                backbone_save_pth = os.path.join(fmap_pth, f'up/block_{idx}/backbone/')
                backbone_fourier_pth = os.path.join(fmap_pth, f'up/block_{idx}/fourier/')            
                skip_save_pth = os.path.join(fmap_pth, f'up/block_{idx}/skip/')

                s_sample_viz = res_samples[-1]


                ret_freq, skip_freq = save_fmap(sample,  backbone_save_pth, skip_sample=s_sample_viz, skip_save_pth=skip_save_pth, time_idx=time_idx, fourier_pth=backbone_fourier_pth)
                
                if segment_masks is not None:
                    cal_ious(sample, segment_masks, save_pth=backbone_save_pth, time_idx=time_idx, threshold_r=iou_threshold)
                    cal_ious(s_sample_viz, segment_masks, save_pth=skip_save_pth, time_idx=time_idx, threshold_r=iou_threshold)

                for i in range(4):
                    channel_quad_backbone_pth = os.path.join(fmap_pth, f'up/block_{idx}/channel_quad_{i}/backbone/')
                    channel_quad_skip_pth = os.path.join(fmap_pth, f'up/block_{idx}/channel_quad_{i}/skip/')

                    num_channels = sample.shape[1]
                    unit_channel = num_channels // 4
                    save_fmap(sample[:, i*unit_channel:(i+1)*unit_channel], channel_quad_backbone_pth, skip_sample=s_sample_viz[:, i*unit_channel:(i+1)*unit_channel], skip_save_pth=channel_quad_skip_pth, time_idx=time_idx, fourier_pth=None)
                
                # edge difference
                if idx == 6:
                    (ret_diff, skip_diff), (cur_ret_diff, cur_skip_diff) = edge_diff(sample, current=True, skip_sample=s_sample_viz, time_idx=time_idx)
                    f_info['edge_diff']['up'] = ret_diff
                    f_info['current_edge_diff']['up'] = cur_ret_diff
                    f_info['edge_diff']['skip'] = skip_diff
                    f_info['current_edge_diff']['skip'] = cur_skip_diff

                if mask_pth is not None and channels is not None:
                    channel_plot(sample, backbone_save_pth, time_idx, mask_pth, channels)
                    channel_plot(s_sample_viz, skip_save_pth, time_idx, mask_pth, channels)
                    
            else:
                _, ret_freq = fourier_save(sample.sum(dim=1)[0]) 

                s_sample_viz = res_samples[-1]
                _, skip_freq = fourier_save(s_sample_viz.sum(dim=1)[0])

                # if idx == 6:
                #     ret_diff, skip_diff = edge_diff(sample, skip_sample=s_sample_viz, time_idx=time_idx)
                #     f_info['edge_diff']['up'] = ret_diff
                #     f_info['edge_diff']['skip'] = skip_diff

            #print('sample shape:', sample.shape)
                    
            


            low_len = int(0.8 * len(ret_freq))
            low_freq = np.sum(ret_freq[:low_len]) 
            high_freq = np.sum(ret_freq[low_len:])

            f_info['low'][f'depth_{6-idx}']['up'] = low_freq
            f_info['high'][f'depth_{6-idx}']['up'] = high_freq

            low_len = int(0.8 * len(skip_freq))
            skip_low_freq = np.sum(skip_freq[:low_len]) 
            skip_high_freq = np.sum(skip_freq[low_len:])

            f_info['low'][f'depth_{6-idx}']['skip'] = skip_low_freq
            f_info['high'][f'depth_{6-idx}']['skip'] = skip_high_freq


            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample, abs_norm_lists = upsample_block(sample, res_samples, emb, skip_sample, skip_scale=1.0)

                s, b, f = abs_norm_lists
                s_list.append(s)
                b_list.append(b)
                f_list.append(f)               
                
            else:
                sample = upsample_block(sample, res_samples, emb)
                print('no skip conv')

            

        if fmap_pth:
            backbone_save_pth = os.path.join(fmap_pth, f'up/block_7/backbone/')
            backbone_fourier_pth = os.path.join(fmap_pth, f'up/block_7/backbone/fourier/')
            save_fmap(sample, backbone_save_pth, time_idx=time_idx, fourier_pth=backbone_fourier_pth)
            
            f_info['edge_diff']['up'] = ret_diff
            f_info['edge_diff']['skip'] = skip_diff

    
        
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)




        if skip_sample is not None:
            sample += skip_sample

        #sample.requires_grad_(True)
        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps



        #sample.requires_grad_(True)
        if not return_dict:
            return (sample,)
        
            
        
        ## loss function for prediction
        if cal_grad:
            #print('sample0 requires grad:', sample0.requires_grad)
            _, predicted_img = scheduler.step_pred(model_output=sample, timestep=schedule_t, sample=sample0)
            # if not res_samples[0].requires_grad:
            #     res_samples[0].requires_grad_(True)

            #print('predicted require grad:', predicted_img.requires_grad)
            normal_img = torch.load('/home/myeongjin/score_sde_pytorch/sampled_images/low_pass_drop/new_sampling/seed_10/normal/b_scale_1.0_s_scale_1.0_t_1_s_1.0_ns_1200/output_tensor.pt')
            #loss = ((predicted_img[0] - normal_img[0])**2).mean()
            

            
            x_list = [f'down_{i}' for i in range(7)]
            x_list += ['mid']
            x_list += [f'up_{i}' for i in range(7)]

            for i in range(sample.shape[0]):
                save_pth = os.path.join(fmap_pth, f'grad/idx_{i}/t_{time_idx+1}')
                os.makedirs(save_pth, exist_ok=True)

                #loss = ((predicted_img[i] - normal_img[i])**2).mean()
                loss = (sample[i] ** 2).mean()

                loss.backward(retain_graph=True)
                

                y_list = []

                # print('down:')
                # print(len(down_out_list))
                for d_grad in list(reversed(down_out_list))[:7]:
                    #print(d_grad)
                    #print(d_grad[0][1][2])

                    #print('d_output grad shape:', d_grad.shape)
                    #y_list.append(d_grad[i].abs().mean().item())
                    y_list.append(d_grad[i].mean().item())

                #print(mid_in_list)
                #print(len(mid_out_list))
                #y_list.append(mid_out_list[0][i].abs().mean().item())
                y_list.append(mid_out_list[0][i].mean().item())
                    
                # print('up:')
                # print(len(up_out_list))
                for u_grad in list(reversed(up_out_list))[:7]:
                    #print(u_grad)
                    
                    #print('u_output grad shape:', u_grad.shape)
                    #print('u grad shape:', u_grad[i].shape)
                    #y_list.append(u_grad[i].abs().mean().item())
                    y_list.append(u_grad[i].mean().item())

                # for u_input in up_in_list[-7:]:
                #     print('len u input:', len(u_input))
                #     for i in range(len(u_input)):
                        
                #         if u_input[i] != None:
                #             print('i:', i, 'u_input grad shape:', u_input[i].shape) 


                plt.figure(figsize=(10, 5))
                plt.plot(x_list, y_list)

                plt.tight_layout()
                plt.savefig(os.path.join(save_pth, 'grad abs mean.png'))
                

            d_hook.remove()
            m_hook.remove()
            u_hook.remove()
                
            

            
                


        return UNet2DOutput(sample=sample), (s_list, b_list, f_list), f_info
