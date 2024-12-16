# Diffusion Modules Analysis
Exploration and analysis of the roles of individual blocks in Diffusion UNet.

## Description
This repository aims to analyze the roles of each block in the UNet architecture commonly used in diffusion models, across different denoising time steps. We categorize the components of UNet architecture into three types of blocks: down, skip, and up. By scaling the feature maps of each block at specific time steps, we examine how the output image changes. Through this process, we investigate the roles of the blocks (down, skip, up) at different denoising time steps (early, middle, later).

```console
python run_ve.py --seed 10 --bsize 4 \
            --is_save --num_steps 400 \
            --t_start_idx 3 \ # We divide the entire time steps into 10 segments and this arugment indicates from which segment the scaling begins
            --duration 2 \ # The duration for which scaling will be applied, e.g., 0.7T ~ 0.5T (T: the entire time steps)
            -down_ids 0 \ # Down block ID to which scaling will be applied (ranging from 0 to 7 in the case of the VE model)
            -skip_ids 6 \ # Skip block ID to which scaling will be applied (ranging from 0 to 7 in the case of the VE model)
            -up_ids 6 \ # Up block ID to which scaling will be applied (ranging from 0 to 7 in the case of the VE model)
            --scales 1 1 2 # Scale factors for (down block, skip block, up block) 
```


![Diffusion_module_analysis_figure](https://github.com/user-attachments/assets/8770ea79-7610-4eaf-8964-3edd9c8dc90e)
