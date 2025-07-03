import lmdb
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch
from bliva.datasets.datasets.base_dataset import BaseDataset, LmdbDataset
import os
import json
import PIL
from PIL import Image
import numpy as np
# from torchvision.transforms.functional import resize, to_tensor
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import resize, to_tensor

class MJ_ST_Dataset(LmdbDataset):
    def __init__(self, vis_processor, text_processor, lmdb_paths, fixed_prompt="What is the text in the image?"):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.fixed_prompt = fixed_prompt
        self.lmdb_envs = [lmdb.open(path, readonly=True, lock=False) for path in lmdb_paths]
        self.transform = ResizeNormalize(size=(224, 224))

        # 获取数据集大小
        self.length = sum([int(env.stat()['entries']) for env in self.lmdb_envs])

    def __getitem__(self, index):
        lmdb_index = index
        env_index = 0

        # 确定从哪个LMDB数据库读取数据
        for i, env in enumerate(self.lmdb_envs):
            num_entries = int(env.stat()['entries'])
            if lmdb_index < num_entries:
                env_index = i
                break
            lmdb_index -= num_entries

        # 从对应的 LMDB 读取数据
        with self.lmdb_envs[env_index].begin(write=False) as txn:
            image_key = f'image-{lmdb_index}'
            label_key = f'label-{lmdb_index}'

            # 读取图像和标签
            image_bytes = txn.get(image_key.encode())
            label = txn.get(label_key.encode()).decode('utf-8')
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 处理图像
        image = self.transform(image).to(dtype=torch.float16)
        
        # 处理文本输入和输出
        text_input = self.text_processor(self.fixed_prompt)
        text_output = label

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
        }

    def __len__(self):
        return self.length

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
        }

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        return img
