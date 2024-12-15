
CUDA_VISIBLE_DEVICES=2 python run_ve.py --seed 10 --bsize 4 \
            --is_save --num_steps 400 --t_start_idx 3 --duration 2 -down_ids 0 -skip_ids 6 -up_ids 6 --scales 1 1 2 -fmap
            
    




