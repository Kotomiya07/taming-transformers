#python scripts/create_eval_data.py data/celebahq/data256x256 data/eval_data/celebahq data/celebahqvalidation.txt

#python scripts/reconstruct_first_stages.py --config models/first_stage_models/kl-f2/config.yaml --ckpt models/first_stage_models/kl-f2/model.ckpt --input_dir data/eval_data/cifar-10 --output_dir reconstructed_images_pretrain/cifar-10-kl-f2 --image_size 32

echo "epoch 20" >> logs/2024-11-05T20-12-32_lsun/eval.txt
python scripts/reconstruct_first_stages.py --config models/first_stage_models/vq-f8/config.yaml --ckpt models/first_stage_models/vq-f8/model.ckpt --input_dir data/eval_data/lsun --output_dir reconstructed_images_pretrain/lsun-vq-f8 --image_size 256
python scripts/reconstruct_first_stages.py --config logs/2024-11-05T20-12-32_lsun/configs/2024-11-08T02-35-33-project.yaml --ckpt logs/2024-11-05T20-12-32_lsun/checkpoints/epoch=000020.ckpt --input_dir data/eval_data/lsun --output_dir reconstructed_images_train/lsun-020 --image_size 256

python scripts/evaluate_first_stages.py --original_dir data/eval_data/lsun --reconstructed_dir1 reconstructed_images_pretrain/lsun-vq-f8 --reconstructed_dir2 reconstructed_images_train/lsun-020 >> logs/2024-11-05T20-12-32_lsun/eval.txt

python scripts/reconstruct_first_stages.py --config logs/2024-11-05T20-12-32_lsun/configs/2024-11-08T02-35-33-project.yaml --ckpt logs/2024-11-05T20-12-32_lsun/checkpoints/epoch=000021.ckpt --input_dir data/eval_data/lsun --output_dir reconstructed_images_train/lsun-021 --image_size 256

echo "epoch 21" >> logs/2024-11-05T20-12-32_lsun/eval.txt
python scripts/evaluate_first_stages.py --original_dir data/eval_data/lsun --reconstructed_dir1 reconstructed_images_pretrain/lsun-vq-f8 --reconstructed_dir2 reconstructed_images_train/lsun-021 >> logs/2024-11-05T20-12-32_lsun/eval.txt

python scripts/reconstruct_first_stages.py --config logs/2024-11-05T20-12-32_lsun/configs/2024-11-08T02-35-33-project.yaml --ckpt logs/2024-11-05T20-12-32_lsun/checkpoints/epoch=000023.ckpt --input_dir data/eval_data/lsun --output_dir reconstructed_images_train/lsun-023 --image_size 256

echo "epoch 23" >> logs/2024-11-05T20-12-32_lsun/eval.txt
python scripts/evaluate_first_stages.py --original_dir data/eval_data/lsun --reconstructed_dir1 reconstructed_images_pretrain/lsun-vq-f8 --reconstructed_dir2 reconstructed_images_train/lsun-023 >> logs/2024-11-05T20-12-32_lsun/eval.txt