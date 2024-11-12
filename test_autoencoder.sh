#python scripts/create_eval_data.py data/celebahq/data256x256 data/eval_data/celebahq data/celebahqvalidation.txt

#python scripts/reconstruct_first_stages.py --config models/first_stage_models/kl-f2/config.yaml --ckpt models/first_stage_models/kl-f2/model.ckpt --input_dir data/eval_data/cifar-10 --output_dir reconstructed_images_pretrain/cifar-10-kl-f2 --image_size 32

echo "epoch 21" >> logs/2024-11-08T13-33-52_celeba/eval.txt
python scripts/reconstruct_first_stages.py --config logs/2024-11-08T13-33-52_celeba/configs/2024-11-08T13-33-52-project.yaml --ckpt logs/2024-11-08T13-33-52_celeba/checkpoints/epoch=000021.ckpt --input_dir data/eval_data/celebahq --output_dir reconstructed_images_train/celebahq-021 --image_size 256

python scripts/evaluate_first_stages.py --original_dir data/eval_data/celebahq --reconstructed_dir1 reconstructed_images_pretrain/celebahq-vq-f4 --reconstructed_dir2 reconstructed_images_train/celebahq-021 >> logs/2024-11-08T13-33-52_celeba/eval.txt

#python scripts/reconstruct_first_stages.py --config logs/2024-11-10T04-50-45_cifar10/configs/2024-11-10T04-50-45-project.yaml --ckpt logs/2024-11-10T04-50-45_cifar10/checkpoints/epoch=000075.ckpt --input_dir data/eval_data/cifar-10 --output_dir reconstructed_images_train/cifar-10-075 --image_size 32

#echo "epoch 75" >> logs/2024-11-10T04-50-45_cifar10/eval.txt
#python scripts/evaluate_first_stages.py --original_dir data/eval_data/cifar-10 --reconstructed_dir1 reconstructed_images_pretrain/cifar-10-kl-f2 --reconstructed_dir2 reconstructed_images_train/cifar-10-075 >> logs/2024-11-10T04-50-45_cifar10/eval.txt

#python scripts/reconstruct_first_stages.py --config logs/2024-11-10T04-50-45_cifar10/configs/2024-11-10T04-50-45-project.yaml --ckpt logs/2024-11-10T04-50-45_cifar10/checkpoints/epoch=000078.ckpt --input_dir data/eval_data/cifar-10 --output_dir reconstructed_images_train/cifar-10-078 --image_size 32

#echo "epoch 78" >> logs/2024-11-10T04-50-45_cifar10/eval.txt
#python scripts/evaluate_first_stages.py --original_dir data/eval_data/cifar-10 --reconstructed_dir1 reconstructed_images_pretrain/cifar-10-kl-f2 --reconstructed_dir2 reconstructed_images_train/cifar-10-078 >> logs/2024-11-10T04-50-45_cifar10/eval.txt

#echo "epoch 41" >> logs/2024-11-10T05-01-00_cifar10/eval.txt
#python scripts/reconstruct_first_stages.py --config logs/2024-11-10T05-01-00_cifar10/configs/2024-11-10T05-01-00-project.yaml --ckpt logs/2024-11-10T05-01-00_cifar10/checkpoints/epoch=000041.ckpt --input_dir data/eval_data/cifar-10 --output_dir reconstructed_images_train/cifar-10-041 --image_size 32
#python scripts/evaluate_first_stages.py --original_dir data/eval_data/cifar-10 --reconstructed_dir1 reconstructed_images_pretrain/cifar-10-kl-f2 --reconstructed_dir2 reconstructed_images_train/cifar10-041 >> logs/2024-11-10T05-01-00_cifar10/eval.txt