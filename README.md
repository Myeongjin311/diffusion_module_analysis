# Diffusion Modules Analysis
Exploration and analysis of the roles of individual blocks in Diffusion UNet.

## Description
This repository aims to analyze the roles of each block in the UNet architecture commonly used in diffusion models, across different denoising time steps. We categorize the components of UNet architecture into three types of blocks: down, skip, and up. By scaling the feature maps of each block at specific time steps, we examine how the output image changes. Through this process, we investigate the roles of the blocks (down, skip, up) at different denoising time steps (early, middle, later).

'''
python run_ve.py
'''


![Diffusion_module_analysis_figure](https://github.com/user-attachments/assets/8770ea79-7610-4eaf-8964-3edd9c8dc90e)
