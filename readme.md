<!-- VAE -->

accelerate launch --config_file accelerate_config.yaml train.py --config ./configs/config_auto.yaml --save_dir /scratch/rgoel15/motion_emitation_data/model_image_diff_z_vae --model vae --data_type simple --wandb_name image_diff_z_vae

<!-- SAM -->

accelerate launch --config_file accelerate_config.yaml train.py --config ./configs/config_auto.yaml --save_dir /scratch/rgoel15/motion_emitation_data/model_image_diff_z_with_masked_inputs --model masked_inputs --data_type simple --wandb_name image_diff_z_with_masked_inputs

<!-- I2-I1 -->

accelerate launch --config_file accelerate_config.yaml train.py --config ./configs/config_auto.yaml --save_dir /scratch/rgoel15/motion_emitation_data/model_image_difference --model autoencoder --data_type simple --wandb_name image_difference
