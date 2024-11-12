import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from torchvision import transforms
from PIL import Image


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, flip_p=0.5):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
    
    def __getitem__(self, i):
        example = self.data[i]
        image = example["image"]

        # Convert image from range -1~1 to 0~255
        image = image * 127.5 + 127.5
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example



class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, flip_p=0.5):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __getitem__(self, i):
        example = self.data[i]
        image = example["image"]
        # Convert image from range -1~1 to 0~255
        image = image * 127.5 + 127.5
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


