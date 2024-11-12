import lmdb
import numpy as np
from PIL import Image
import io
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def save_image(key_value, output_dir, index):
    key, value = key_value
    image = Image.open(io.BytesIO(value))
    image.save(f"{output_dir}/{index:06d}.png")

def lmdb_to_images(lmdb_path, output_dir, num_workers=4):
    print(f"Converting {lmdb_path} to {output_dir} with {num_workers} workers")
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(save_image, (key, value), output_dir, i) for i, (key, value) in enumerate(cursor)]
            for i, future in enumerate(as_completed(futures)):
                if i % 1000 == 0 and i > 0:
                    print(f"{i} images processed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', type=str, default="./lsun/church_outdoor_train_lmdb")
    parser.add_argument('--output_dir', type=str, default="./lsun/churches/train")
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker threads')
    args = parser.parse_args()

    lmdb_path = args.lmdb_path
    output_dir = args.output_dir
    num_workers = args.num_workers

    os.makedirs(output_dir, exist_ok=True)
    lmdb_to_images(lmdb_path, output_dir, num_workers)