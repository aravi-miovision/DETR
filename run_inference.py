import argparse
import numpy as np
import torch
import util.misc as utils

from datasets import build_dataset
from models import build_model
from main import get_args_parser
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

font = ImageFont.truetype("/mnt/storage/aravi/hackday_sep22/arial.ttf", 16)
CLASS_INDICES = [1, 2, 3, 4, 6, 8]

def get_labels():
    with open("labels.txt", "r") as f:
        labels = f.readlines()
    labels = [l.replace('\n', "") for l in labels]
    return ['background'] + labels

gt_labels = get_labels()

def annotate_and_save(file_path, image_array, scores, class_names, boxes):
    image_array = np.moveaxis((image_array * 255.).astype(np.uint8), 0, -1)
    pil_img = Image.fromarray(image_array)
    drw_img = ImageDraw.Draw(pil_img)
    for (score, class_name, bbox) in zip(scores.squeeze(), class_names, boxes.squeeze()):
        if gt_labels.index(class_name) not in CLASS_INDICES:
            continue
        conf = "%d"%(score*100)
        drw_img.text(
            (bbox[0], bbox[1]),f"{class_name} {conf}%",(255,255,255),font=font
        )
        drw_img.rectangle(xy=bbox, width=1)
    pil_img.save(file_path)

class UnNormalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def infer(args, model_path, data_loader, threshold=0.5, max_images = 500):
    un_normalize = UnNormalize()
    checkpoint = torch.load(model_path, map_location='cpu')
    model, criterion, postprocessors = build_model(args)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    torch.save(model, "ttest.pth")
    breakpoint()

    count = 0
    for (samples, targets) in data_loader:
        outputs = model(samples)
        orig_target_sizes = torch.stack([torch.tensor([384, 600]) for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        for idx in tqdm(range(len(results))):
            try:
                img = un_normalize(samples.tensors[idx]).numpy()
                scores, labels, boxes = results[idx]['scores'].numpy(), results[idx]['labels'].numpy(), results[idx]['boxes'].numpy()
                if scores.size == 0:
                    continue
                selection = np.argwhere(scores >= threshold)
                th_scores = scores[selection]
                th_labels = labels[selection]
                th_boxes = boxes[selection]
                th_classes = [gt_labels[idx] for idx in th_labels.squeeze()]
                annotate_and_save(f"/mnt/storage/aravi/hackday_sep22/samples/{count}_{idx}.jpeg", img, th_scores, th_classes, th_boxes)
            except:
                continue
        count += 1

        if count*samples.tensors.shape[0] > max_images:
            break


if __name__=='__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.dataset_file = 'coco'


    dataset_val = build_dataset(image_set='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 2, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    infer(
        args=args,
        # model_path="new_model.pth",
        model_path="/mnt/storage/aravi/hackday_sep22/pretrained_coco/detr-r50-e632da11.pth",
        data_loader = data_loader_val
    )