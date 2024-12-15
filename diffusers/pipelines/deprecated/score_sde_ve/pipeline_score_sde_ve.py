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

from typing import List, Optional, Tuple, Union

import torch

from ....models import UNet2DModel
from ....schedulers import ScoreSdeVeScheduler
from ....utils.torch_utils import randn_tensor
from ...pipeline_utils import DiffusionPipeline, ImagePipelineOutput

import torch.fft as fft
from PIL import Image

import numpy as np
import os

from collections import defaultdict
import matplotlib.pyplot as plt


def fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered





class ScoreSdeVePipeline(DiffusionPipeline):
    r"""
    Pipeline for unconditional image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image.
        scheduler ([`ScoreSdeVeScheduler`]):
            A `ScoreSdeVeScheduler` to be used in combination with `unet` to denoise the encoded image.
    """

    unet: UNet2DModel
    scheduler: ScoreSdeVeScheduler

    def __init__(self, unet: UNet2DModel, scheduler: ScoreSdeVeScheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        
    def save_images(self, i, dir_name, sample_mean_, predicted_, residual_=None):
                
            
        #dir_name_ = os.path.join(dir_name, f'b_scale_{backbone_scale:.2f}_s_scale_{skip_scale:.2f}_ns_{num_inference_steps}')
        predicted_dir = os.path.join(dir_name, 'predicted')
        
        # if pair:
        #     dir_name_ = os.path.join(dir_name_, f'b_scale_{1.0}_s_scale_{(1 - 0.3 * (backbone_scale-1)):.2f}_ns_{num_inference_steps}')
        #     predicted_dir_ = os.path.join(dir_name_, 'predicted')


        os.makedirs(predicted_dir, exist_ok=True)

        savename = f'image_{i+1}.png'
        predicted_name = f'predicted_{i+1}.png'
        savepth = os.path.join(dir_name, savename)
        predicted_pth = os.path.join(predicted_dir, predicted_name)

        # if os.path.exists(savepth) and os.path.exists(predicted_pth):
        #     return

        sample_img_ = sample_mean_.clone()
        sample_img_ = sample_img_.clamp(0, 1)
        sample_img_ = sample_img_.cpu().detach().permute(0,2,3,1)

        
        predicted_img_ = predicted_.clamp(0, 1)
        predicted_img_ = predicted_img_.cpu().detach().permute(0,2,3,1)
        
        sample_img_ = torch.cat([torch.cat([sample_img_[0], sample_img_[1]], dim=0), torch.cat([sample_img_[2], sample_img_[3]], dim=0)], dim=1).unsqueeze(0)
        predicted_img_ = torch.cat([torch.cat([predicted_img_[0], predicted_img_[1]], dim=0), torch.cat([predicted_img_[2], predicted_img_[3]], dim=0)], dim=1).unsqueeze(0)
        
        sample_img_ = sample_img_.numpy()
        predicted_img_ = predicted_img_.numpy()

        ret = sample_img_
        
        sample_img_ = self.numpy_to_pil(sample_img_)
        predicted_img_ = self.numpy_to_pil(predicted_img_)
        
        
        
        sample_img_[0].save(savepth)
        predicted_img_[0].save(predicted_pth)

        if residual_ is not None:
            residual_dir = os.path.join(dir_name, 'residual')
            os.makedirs(residual_dir, exist_ok=True)
            residual_pth = os.path.join(residual_dir, f'residual_{i+1}.png')

            residual_img_ = residual_.clamp(0, 1)
            residual_img_ = residual_img_.cpu().detach().permute(0,2,3,1)
            residual_img_ = torch.cat([torch.cat([residual_img_[0], residual_img_[1]], dim=0), torch.cat([residual_img_[2], residual_img_[3]], dim=0)], dim=1).unsqueeze(0)
            residual_img_ = residual_img_.numpy()
            residual_img_ = self.numpy_to_pil(residual_img_)
            residual_img_[0].save(residual_pth)

        return ret

    @torch.no_grad()
    def __call__(
        self,
        dir_name: str = None,
        batch_size: int = 1,
        num_inference_steps: int = 2000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        seed: int = 0,
        threshold: int = 1,
        scale: float = 0.9,
        is_save: bool = True,
        scales: list = None,
        normalized_scale: bool = False,
        t_start_idx: float = None,
        select_channels: list = None,
        channel_inverse: bool = False,
        gradients: bool = False,
        block_indices: list = None,
        duration: int = None,
        relation_check: bool = False,
        feature_map_viz: bool = False,
        mask_pth: str = None,
        channels: list = None,
        segment_masks: dict = None,
        iou_threshold: float = 0.8,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, `optional`):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            output_type (`str`, `optional`, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """


        low_freq_trackers = dict()
        high_freq_trackers = dict()

        # edge_diffs = defaultdict(list)
        # current_edge_diffs = defaultdict(list)
        for i in range(7):
            low_freq_trackers[f'depth_{i}'] = defaultdict(list)
            high_freq_trackers[f'depth_{i}'] = defaultdict(list)



        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        sample = randn_tensor(shape, generator=generator) * self.scheduler.init_noise_sigma
        sample = sample.to(self.device)
        
        info_pth = os.path.join(dir_name, 'info')    
        os.makedirs(info_pth, exist_ok=True)
        dynamics_txt_pth = os.path.join(dir_name, f'info/dynamics.txt')
        dynamics_f = open(dynamics_txt_pth, 'w') 
         
        predicted_dir = os.path.join(dir_name, 'predicted')
        os.makedirs(predicted_dir, exist_ok=True)            

        

        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.set_sigmas(num_inference_steps)
        

        total_s_list = []
        total_b_list = []
        total_f_list = []
        
        total_output_list = []
        #fourier_output_list = []
        
        t_unit = num_inference_steps // 10
        t_start = t_start_idx * t_unit
        t_end = t_start + duration * t_unit

        print('t_unit:', t_unit)
        print('t_start:', t_start)
        print('t_end:', t_end)

        


        difference_dict = defaultdict(list)
        
        ## create fmap dirs
        if feature_map_viz:
            fmap_pth = os.path.join(dir_name, 'fmap_viz/')
            
            for i in range(7):
                os.makedirs(os.path.join(fmap_pth, f'down/block_{i}'), exist_ok=True)
                os.makedirs(os.path.join(fmap_pth, f'up/block_{i}/backbone'), exist_ok=True)
                os.makedirs(os.path.join(fmap_pth, f'up/block_{i}/skip'), exist_ok=True)
        
        if mask_pth is not None:
            mask = torch.load(mask_pth)
            mask = torch.from_numpy(mask).to(self.device)
            print('mask shape:', mask.shape)
        else:
            mask = None

        # iterative denoising steps
        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):

            sigma_t = self.scheduler.sigmas[i] * torch.ones(shape[0], device=self.device)
            

                
            if gradients and (i % 120 == 0):
                cal_grad = True
            else:
                cal_grad = False
            
            fmap_pth_ = None
         
            if (i + 1) % 40 == 0 and feature_map_viz:
                fmap_pth_ = fmap_pth
            
            
            for _ in range(self.scheduler.config.correct_steps):
                #model_output = self.unet(sample, sigma_t, time_idx=i).sample
                
                if (i == t_start) and relation_check:
                    sample1 = sample.clone()
                    sample2 = sample.clone()

                if t_start <= i and i < t_end:
                    if cal_grad:
                        with torch.enable_grad():
                            model_output, norm_lists = self.unet(sample, sigma_t, time_idx=i, seed=seed, threshold=threshold, scale=scale, scales=scales, \
                                                            normalized_scale=normalized_scale, select_channels=select_channels, \
                                                            block_indices=block_indices, cal_grad=cal_grad, scheduler=self.scheduler, schedule_t=t, \
                                                                num_inference_steps=num_inference_steps)
                            
                    else:
                        if not relation_check:
                            model_output = self.unet(sample, sigma_t, time_idx=i, \
                                block_indices=block_indices, scales=scales, mask=mask)[0].sample

                        else:
                            model_output = self.unet(sample, sigma_t)[0].sample
                                  
                            model_output1 = self.unet(sample1, sigma_t, block_indices=block_indices, \
                                scales=[1.0, 1.0, scales[2]])[0].sample
                            
                            model_output2 = self.unet(sample2, sigma_t, block_indices=block_indices, \
                                scales=[1.0, scales[1], 1.0])[0].sample
                            
                        
                    
                else:
                    if cal_grad:
                        with torch.enable_grad():
                            model_output, norm_lists = self.unet(sample, sigma_t, time_idx=i, seed=seed, threshold=threshold, scale=scale, scales=scales, \
                                                        block_indices=block_indices, cal_grad=cal_grad, scheduler=self.scheduler, \
                                                            schedule_t=t, num_inference_steps=num_inference_steps)
                    else:
                        if not relation_check:
                            model_output = self.unet(sample, sigma_t, time_idx=i)[0].sample
                        else:
                            model_output = self.unet(sample, sigma_t)[0].sample
                      
                            if i >= t_end:
                                model_output1 = self.unet(sample1, sigma_t)[0].sample                 
                                model_output2 = self.unet(sample2, sigma_t)[0].sample
                   
                
                
                # s_list, b_list, f_list = norm_lists # each element of a list shape: (B, the number of resnet(3)), list length: the number of blocks (7)
                
                # s_tensor = torch.stack(s_list, dim=1)
                # b_tensor = torch.stack(b_list, dim=1)
                # f_tensor = torch.stack(f_list, dim=1) 
                 
                # total_s_list.append(s_tensor)
                # total_b_list.append(b_tensor)
                # total_f_list.append(f_tensor)
                
                
   
                
                
                
                total_output_list.append(model_output.abs().mean(dim=(1,2,3)))

                sample, noise = self.scheduler.step_correct(model_output, sample, generator=generator)
                sample = sample.prev_sample
                
                if (i >= t_start) and relation_check:
                    sample1 = self.scheduler.step_correct(model_output1, sample1, generator=generator, noise=noise)[0].prev_sample
                    sample2 = self.scheduler.step_correct(model_output2, sample2, generator=generator, noise=noise)[0].prev_sample

            # prediction step
            if t_start <= i and i < t_end:
                if not relation_check:
                    model_output, _, f_info = self.unet(sample, sigma_t, time_idx=i, block_indices=block_indices, \
                                scales=scales, fmap_pth=fmap_pth_, mask=mask, mask_pth=mask_pth, channels=channels, \
                                    segment_masks=segment_masks, iou_threshold=iou_threshold)
                    
                    model_output = model_output.sample
                else:
                    
                    model_output = self.unet(sample, sigma_t)[0].sample
                      
                    model_output1 = self.unet(sample1, sigma_t, block_indices=block_indices, \
                        scales=[1.0, 1.0, scales[2]])[0].sample
                    
                    model_output2 = self.unet(sample2, sigma_t, block_indices=block_indices, \
                        scales=[1.0, scales[1], 1.0])[0].sample
                    
                
            else:
                model_output, _, f_info = model(sample, sigma_t, time_idx=i, fmap_pth=fmap_pth_, mask_pth=mask_pth, channels=channels, \
                                                segment_masks=segment_masks, iou_threshold=iou_threshold)

                model_output = model_output.sample
                

                
                if (i >= t_end) and relation_check:
                
                    model_output1 = self.unet(sample1, sigma_t)[0].sample
                    model_output2 = self.unet(sample2, sigma_t)[0].sample
                
            if feature_map_viz:
                for idx in range(7):
                    low_freq_trackers[f'depth_{idx}']['down'].append(f_info['low'][f'depth_{idx}']['down'])
                    low_freq_trackers[f'depth_{idx}']['skip'].append(f_info['low'][f'depth_{idx}']['skip'])
                    low_freq_trackers[f'depth_{idx}']['up'].append(f_info['low'][f'depth_{idx}']['up'])

                    high_freq_trackers[f'depth_{idx}']['down'].append(f_info['high'][f'depth_{idx}']['down'])
                    high_freq_trackers[f'depth_{idx}']['skip'].append(f_info['high'][f'depth_{idx}']['skip'])
                    high_freq_trackers[f'depth_{idx}']['up'].append(f_info['high'][f'depth_{idx}']['up'])

            # edge diff tracker
            # edge_diffs['down'].append(f_info['edge_diff']['down'])
            # edge_diffs['skip'].append(f_info['edge_diff']['skip'])
            # edge_diffs['up'].append(f_info['edge_diff']['up'])

            # current edge diff tracker
            # if fmap_pth_:
            #     current_edge_diffs['down'].append(f_info['current_edge_diff']['down'])
            #     current_edge_diffs['skip'].append(f_info['current_edge_diff']['skip'])
            #     current_edge_diffs['up'].append(f_info['current_edge_diff']['up'])

            
            output, predicted, noise = self.scheduler.step_pred(model_output, t, sample, generator=generator)



            sample, sample_mean = output.prev_sample, output.prev_sample_mean
            
            if (i >= t_start) and relation_check:
                output1, predicted1, _ = self.scheduler.step_pred(model_output1, t, sample1, generator=generator, noise=noise)
                sample1, sample_mean1 = output1.prev_sample, output1.prev_sample_mean
                
                output2, predicted2, _ = self.scheduler.step_pred(model_output2, t, sample2, generator=generator, noise=noise)
                sample2, sample_mean2 = output2.prev_sample, output2.prev_sample_mean
                
                abs_diff_ori_1 = (sample - sample1).abs().mean(dim=(1,2,3))
                abs_diff_ori_2 = (sample - sample2).abs().mean(dim=(1,2,3))
                abs_diff_1_2 = (sample1 - sample2).abs().mean(dim=(1,2,3))
                mean_1_2 = (sample1 - sample2).mean(dim=(1,2,3))
                std_1_2 = (sample1 - sample2).std(dim=(1,2,3))
                
                maps_1_2 = (sample1 - sample2).reshape(batch_size, -1)
                
                dynamics_f.write(f'time idx: {i}\n')
                dynamics_f.write(f'diff_ori_1: {abs_diff_ori_1}\n')
                dynamics_f.write(f'diff_ori_2: {abs_diff_ori_2}\n')
                dynamics_f.write(f'diff_1_2: {abs_diff_1_2}\n')
                dynamics_f.write('\n')
                
                difference_dict['ori_1'].append(abs_diff_ori_1.detach().cpu())
                difference_dict['ori_2'].append(abs_diff_ori_2.detach().cpu())
                difference_dict['1_2'].append(abs_diff_1_2.detach().cpu())
                difference_dict['mean_1_2'].append(mean_1_2.detach().cpu())
                difference_dict['std_1_2'].append(std_1_2.detach().cpu())
                difference_dict['maps_1_2'].append(maps_1_2.detach().cpu())
            

            if (i+1) % 40 == 0 and is_save:
                
                
                if (i >= t_start) and relation_check:
                    ret_ori = self.save_images(i, dir_name, sample_mean, predicted)
                    ret1 = self.save_images(i, dir_name, sample_mean1, predicted1)
                    ret2 = self.save_images(i, dir_name, sample_mean2, predicted2)
  
                elif not relation_check:
                    self.save_images(i, dir_name, sample_mean, predicted)
                


        
        # last_tensor_pth = os.path.join(dir_name, 'output_tensor.pt')
        # torch.save(sample_mean, last_tensor_pth)
        
        sample = sample_mean.clamp(0, 1)
        sample = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            sample = self.numpy_to_pil(sample)

        # edge diff pth
        edge_pth = os.path.join(fmap_pth, 'edge_diff')
        os.makedirs(edge_pth, exist_ok=True)
        
        # edge diff plot
        # timesteps = range(num_inference_steps)
        # plt.plot(timesteps, edge_diffs['down'], label='down')
        # plt.plot(timesteps, edge_diffs['skip'], label='skip')
        # plt.plot(timesteps, edge_diffs['up'], label='up')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(edge_pth, 'edge_diff.png'))

        
        if feature_map_viz:
            # frequency plot pth
            tracker_plot_pth = os.path.join(fmap_pth, 'freq_time')
            
            # frequency plot
            for i in range(7):
                block_pth = os.path.join(tracker_plot_pth, f'depth_{i}')
                os.makedirs(block_pth, exist_ok=True)

                plt.clf()
                
                plt.plot(low_freq_trackers[f'depth_{i}']['down'], label='down')
                plt.plot(low_freq_trackers[f'depth_{i}']['skip'], label='skip')
                plt.plot(low_freq_trackers[f'depth_{i}']['up'], label='up')

                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(block_pth, 'low_frequency.png'))

                plt.clf()

                plt.plot(high_freq_trackers[f'depth_{i}']['down'], label='down')
                plt.plot(high_freq_trackers[f'depth_{i}']['skip'], label='skip')
                plt.plot(high_freq_trackers[f'depth_{i}']['up'], label='up')

                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(block_pth, 'high_frequency.png'))

            plt.clf()

            
            # current edge diff plot
            # current_timesteps = range(t_unit, num_inference_steps+1, t_unit)
            # plt.plot(current_timesteps, current_edge_diffs['down'], label='down')
            # plt.plot(current_timesteps, current_edge_diffs['skip'], label='skip')
            # plt.plot(current_timesteps, current_edge_diffs['up'], label='up')

            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(os.path.join(edge_pth, 'current_edge_diff.png'))
        
        # total_s = torch.cat(total_s_list, dim=-1)
        # total_b = torch.cat(total_b_list, dim=-1)
        # total_f = torch.cat(total_f_list, dim=-1)
        
        total_o = torch.stack(total_output_list, dim=1)
        
        
        #img_dynamics_f.close()
        
        # for idx in range(total_s.shape[1]):
        #     os.makedirs(os.path.join(norm_pth, f'skip/block_idx_{idx}'), exist_ok=True)
        #     torch.save(total_s[:, idx], os.path.join(norm_pth, f'skip/block_idx_{idx}/total.pt'))
        
        # for idx in range(total_b.shape[1]):
        #     os.makedirs(os.path.join(norm_pth, f'backbone/block_idx_{idx}'), exist_ok=True)
        #     torch.save(total_b[:, idx], os.path.join(norm_pth, f'backbone/block_idx_{idx}/total.pt'))
            
        # for idx in range(total_f.shape[1]):
        #     os.makedirs(os.path.join(norm_pth, f'fusion/block_idx_{idx}'), exist_ok=True)
        #     torch.save(total_f[:, idx], os.path.join(norm_pth, f'fusion/block_idx_{idx}/total.pt'))
        
        # print('outputs of unet shape:', total_o.shape)
        # print('sigmas shape:', self.scheduler.sigmas.shape)
        
        #torch.save(total_o, os.path.join(norm_pth, 'unet_outputs.pt'))
        #torch.save(self.scheduler.sigmas, os.path.join(norm_pth, 'sigmas.pt'))
        
        if relation_check:
            dynamics_f.close()
            
            t_list = range(t_start, num_inference_steps)
        
            ori_1 = torch.stack(difference_dict['ori_1'], dim=-1).detach().cpu()
            ori_2 = torch.stack(difference_dict['ori_2'], dim=-1).detach().cpu()
            diff_1_2 = torch.stack(difference_dict['1_2'], dim=-1).detach().cpu()
            mean_1_2 = torch.stack(difference_dict['mean_1_2'], dim=-1).detach().cpu()
            std_1_2 = torch.stack(difference_dict['std_1_2'], dim=-1).detach().cpu()
            
            maps_1_2 = torch.cat(difference_dict['maps_1_2'], dim=-1).detach().cpu()
            mean_1_2_scalar = maps_1_2.mean(dim=-1).detach().cpu()
            std_1_2_scalar = maps_1_2.std(dim=-1).detach().cpu()
            
            info_pth = os.path.join(dir_name, 'info')
            #os.makedirs(os.path.join(info_pth, 'global'), exist_ok=True)
            
            for i in range(batch_size + 1):
                if i != batch_size:
                    dir_name_ = i
                else:
                    dir_name_ = 'global'
                
                os.makedirs(os.path.join(info_pth, f'{dir_name_}'), exist_ok=True)
                
                
                ori_1_pth = os.path.join(info_pth, f'{dir_name_}/ori_1_dynamics.png')
                if i != batch_size:
                    plt.plot(t_list, ori_1[i], label='diff_ori_1')
                else:
                    plt.plot(t_list, torch.mean(ori_1, dim=0), label='diff_ori_1')
                
                
                
                ori_2_pth = os.path.join(info_pth, f'{dir_name_}/ori_2_dynamics.png')
                if i != batch_size:                    
                    plt.plot(t_list, ori_2[i], label='diff_ori_2')
                else:
                    plt.plot(t_list, torch.mean(ori_2, dim=0), label='diff_ori_2')
                
                
                
                
                #diff_1_2_pth = os.path.join(info_pth, f'{dir_name_}/diff_1_2_dynamics.png')
                diff_pth = os.path.join(info_pth, f'{dir_name_}/diff_dynamics.png')
                if i != batch_size: 
                    plt.plot(t_list, diff_1_2[i], label='diff_1_2')
                else:
                    plt.plot(t_list, torch.mean(diff_1_2, dim=0), label='diff_1_2')
                plt.xlabel('timesteps')
                plt.ylabel('diff')
                
                plt.legend()
                plt.savefig(diff_pth)
                plt.clf()
                
                
                
                
                mean_1_2_pth = os.path.join(info_pth, f'{dir_name_}/mean_1_2_dynamics.png')
                if i != batch_size: 
                    plt.plot(t_list, mean_1_2[i])
                else:
                    plt.plot(t_list, torch.mean(mean_1_2, dim=0))
                plt.xlabel('timesteps')
                plt.ylabel('diff')
                plt.savefig(mean_1_2_pth)
                plt.clf()
                
                
                
                
                std_1_2_pth = os.path.join(info_pth, f'{dir_name_}/std_1_2_dynamics.png')
                if i != batch_size: 
                    plt.plot(t_list, std_1_2[i])
                else:
                    plt.plot(t_list, torch.mean(std_1_2, dim=0))
                plt.xlabel('timesteps')
                plt.ylabel('diff')
                plt.savefig(std_1_2_pth)
                plt.clf()
                
                
                
                with open(os.path.join(info_pth, f'{dir_name_}/mean_std.txt'), 'w') as f:
                    if i != batch_size: 
                        f.write(f'mean: {mean_1_2_scalar[i].item()}\n')
                        f.write(f'std: {std_1_2_scalar[i].item()}\n')
                    else:
                        f.write(f'mean: {torch.mean(maps_1_2)}\n')
                        f.write(f'std: {torch.std(std_1_2_scalar)}\n')

            
            
        
        return ImagePipelineOutput(images=sample)
