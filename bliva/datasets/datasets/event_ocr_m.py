import torch
from bliva.datasets.datasets.base_dataset import BaseDataset
import os
import json
from PIL import Image
import numpy as np

class EVENTOCRDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, sam_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, sam_root, ann_paths)
        
        # Load annotation from JSON
        self.annotation = []
        for ann_path in ann_paths:
            with open(ann_path, 'r') as f:
                self.annotation.extend(json.load(f))

    def __getitem__(self, index):
        ann = self.annotation[index]

        # Get image path and open image
        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.image_to_tensor(image)
        # sam_image_path = os.path.join(self.sam_root, ann["image"])
        # sam_image = Image.open(sam_image_path).convert("RGB")

        # seed = torch.randint(0, 100000, (1,)).item()
        # # Process image
        # torch.manual_seed(seed)
        # image = self.vis_processor(image) #torch.Size([3, 224, 224])
        # torch.manual_seed(seed)
        # sam_image = self.vis_processor(sam_image)

        # image = image.permute(1, 2, 0).cpu().numpy()
        # image = (image * 255).astype(np.uint8)

        # sam_image = sam_image.permute(1, 2, 0).cpu().numpy()
        # sam_image = (sam_image * 255).astype(np.uint8)

        # # 使用 PIL 保存图片
        # Image.fromarray(image).save("oral_image.jpg")
        # Image.fromarray(sam_image).save("sam_image.jpg") 
        # Extract the question and answer from conversations
        conversations = ann["conversations"]
        boxes = ann["boxes"]
        question = ""
        answer = ""

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.ones((len(boxes),), dtype=torch.int64)  # Assume all boxes are of class 1

        targets = {"boxes": boxes_tensor, "labels": labels_tensor}

        # image, target = self.transforms(img, target)

        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            elif conv["from"] == "gpt":
                answer = conv["value"]

        # Process text input (if needed) and set answer
        text_input = question
        text_output = answer

        return {
            "image": image,
            "image_path":image_path,
            "targets": targets,
            "text_input": text_input,
            "text_output": text_output,
        }

    def collater(self, samples):
        image_list, image_path_list, targets_list, question_list, answer_list = [], [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            image_path_list.append(sample["image_path"])
            targets_list.append(sample["targets"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["text_output"])

        return {
            "image": torch.stack(image_list, dim=0),
            "image_path": image_path_list,
            "targets": targets_list,
            "text_input": question_list,
            "text_output": answer_list,
        }

    def image_to_tensor(self, image):
        # Convert PIL image to a tensor
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image /= 255.0  # Normalize to [0, 1]
        return image