import subprocess

# python scripts/txt2img.py --prompt "a sunset behind a mountain range, vector image" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0  を実行
subprocess.run(["python", "scripts/txt2img.py", "--prompt", "a sunset behind a mountain range, vector image", "--ddim_eta", "1.0", "--n_samples", "1", "--n_iter", "1", "--H", "384", "--W", "1024", "--scale", "5.0"])

