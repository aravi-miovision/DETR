from pathlib import Path
from PIL import Image
import os
import torch
import torch.utils.data
import torchvision
import typing
import torchvision.transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
try:
    from .tf_utils import parse_record
except:
    from tf_utils import parse_record

# import datasets.transforms as T

from tqdm import tqdm

class NautilusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tfrecord_path: str,
        transforms: typing.Optional[typing.Callable] = None,
        is_train: bool = True,
        input_size: tuple = (300, 300),
        buffer_size: int = 512
    ):
        super(NautilusDataset, self).__init__()
        self._tfrecord_path = tfrecord_path
        self._transforms = transforms
        self._is_train = is_train
        self._input_size = input_size
        self._buffer_size = buffer_size

        if self._transforms is None:
            self._transforms = self.get_default_transforms()

        self._dataset = tf.data.TFRecordDataset(self._tfrecord_path)
        self._dataset_iter = iter(self._dataset)

        self.get_new_buffer()

    def get_default_transforms(self):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
            
        if self._is_train:
            return T.Compose([
                T.Resize((self._input_size[0], self._input_size[1])),
                normalize,
            ])
        return normalize

    def get_new_buffer(self):
        self._read_count = 0
        self._buffer = list()
        for i in tqdm(range(self._buffer_size)):
            try:
                record = next(self._dataset_iter)
            except:
                self._dataset_iter = iter(self._dataset)
                record = next(self._dataset_iter)
            record_data = parse_record(record)
            img = self.get_image(record_data)
            img = self._transforms(img)
            target = self.get_targets(record_data)
            self._buffer.append((img, target))

    def get_image(self, record_data):
        return Image.fromarray(record_data.image)

    def get_targets(self, record_data):
        width, height = record_data.width, record_data.height
        boxes = torch.tensor(record_data.boxes)
        target = {
            "boxes": boxes,
            "labels": torch.tensor(record_data.labels, dtype=torch.int64),
            "area": torch.tensor([
                (b[2] * width * b[3] * height) for b in boxes
            ], dtype=torch.float32),
            "orig_size": torch.as_tensor([int(height), int(width)], dtype=torch.int64),
            "size": torch.as_tensor(self._input_size, dtype=torch.int64)
        }
        return target

    def __len__(self):
        return 900000

    def __getitem__(self, idx):
        self._read_count += 1
        if self._read_count >= self._buffer_size:
            self.get_new_buffer()
        return self._buffer[(idx%self._buffer_size)]

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": root / "train.tfrecord",
        "val": root / "valid.tfrecord",
    }
    dataset = NautilusDataset(tfrecord_path=PATHS[image_set])
    return dataset

if __name__=='__main__':
    nd = NautilusDataset("/mnt/storage/2021-09-24-its/train.tfrecord")
    breakpoint()