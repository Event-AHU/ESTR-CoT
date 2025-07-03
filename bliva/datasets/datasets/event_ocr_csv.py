import torch
from bliva.datasets.datasets.base_dataset import BaseDataset
import os
import json
import PIL
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import resize, to_tensor
from torch_geometric.data import Data, Batch
class EVENTOCRDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, csv_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, csv_root, ann_paths)
        
        # Load annotation from JSON
        self.transform = ResizeNormalize(size=(224,224))
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path, 'r') as f:
                self.annotation.extend(json.load(f))

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Get image path and open image
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).to(dtype=torch.float16)

        # Get event data path
        id = ann["id"]
        prefix, suffix = id.split("-", 1) 
        real_path_id = f"{prefix}/{suffix}"
        event_path = os.path.join(self.csv_root, f"{real_path_id}.csv")

        # Process event data
        events = self.extract_first_frame_from_csv(event_path)
        events = self.preprocess_events(events)
        events_data = self.to_data(**events)
        events_data = self.format_data(events_data)

        # Process text data
        conversations = ann["conversations"]
        question, answer = "", ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            elif conv["from"] == "gpt":
                answer = conv["value"]
        question = self.text_processor(question)

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "event_data": events_data,
        }

    def collater(self, samples):
        # Initialize lists to store data
        images, text_inputs, text_outputs, event_data_list = [], [], [], []

        for sample in samples:
            images.append(sample["image"])
            text_inputs.append(sample["text_input"])
            text_outputs.append(sample["text_output"])
            event_data_list.append(sample["event_data"])

        # 使用 Batch.from_data_list 将 event_data_list 转换成批次格式
        combined_event_data = Batch.from_data_list(event_data_list)
        combined_event_data.height = 224
        combined_event_data.width = 224
        combined_event_data.time_window = 1000000
        combined_event_data.num_graphs = 2

        # 创建最终的输出字典
        return {
            "image": torch.stack(images, dim=0),  # Stack images as before
            "text_input": text_inputs,
            "text_output": text_outputs,
            "event_data": combined_event_data,  # Include combined event data in the required format
        }

    def extract_first_frame_from_csv(self, csv_file):
        # 读取 CSV 文件
        df = pd.read_csv(csv_file, header=None, names=["x", "y", "p", "t"])

        # 获取最小和最大时间戳
        t_min = df["t"].min()
        t_max = df["t"].max()

        # 将时间划分为 19 个帧
        frame_intervals = np.linspace(t_min, t_max, 20)  # 生成20个分隔点，19个区间

        # 取第一个帧的时间窗口
        t_start = frame_intervals[0]
        t_end = frame_intervals[1]

        # 筛选第一个时间窗口中的事件
        events_in_first_frame = df[(df["t"] >= t_start) & (df["t"] < t_end)]

        # 将事件数据转为字典格式并返回
        events = {
            'p': events_in_first_frame["p"].values.astype(np.uint8),
            't': events_in_first_frame["t"].values,
            'x': events_in_first_frame["x"].values.astype(np.uint16),
            'y': events_in_first_frame["y"].values.astype(np.uint16)
        }

        return events

    def preprocess_events(self, events):
        mask = events["y"] < 224
        events = {k: v[mask] for k, v in events.items()}
        if len(events["t"]) > 0:
            events["t"] = 1000000 + events["t"] - events["t"][-1]
        events["p"] = 2 * events["p"].reshape((-1, 1)).astype("int8") - 1
        return events

    def to_data(self, **kwargs):
        # convert all tracks to correct format
        xy = np.stack([kwargs['x'], kwargs['y']], axis=-1).astype("int16")
        t = kwargs['t'].astype("int32")
        p = kwargs['p'].reshape((-1,1))

        kwargs['x'] = torch.from_numpy(p)
        kwargs['pos'] = torch.from_numpy(xy)
        kwargs['t'] = torch.from_numpy(t)

        return Data(**kwargs)

    def format_data(self, data, normalizer=None):
        if normalizer is None:
            normalizer = torch.stack([torch.tensor(224), torch.tensor(224), torch.tensor(1000000)], dim=-1)

        data.pos = torch.cat([data.pos, data.t.view((-1,1))], dim=-1)
        data.t = None
        data.x = data.x.float()
        data.pos = data.pos / normalizer
        return data



class ResizeNormalize(object):
    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        # img.sub_(0.5).div_(0.5)
        # img = TF.normalize(img, self.mean, self.std)
        return img