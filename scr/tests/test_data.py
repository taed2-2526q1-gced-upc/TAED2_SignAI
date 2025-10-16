# pip install "deepchecks[vision]" torch pillow matplotlib  # if needed

from deepchecks.vision.suites import train_test_validation # type: ignore
from deepchecks.vision import VisionData, BatchOutputFormat # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from PIL import Image
import numpy as np
import os

class PNGTxtDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.label_dir = label_dir

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        ip = self.image_paths[idx]
        # read image as RGB uint8 [0..255]
        img = Image.open(ip).convert("RGB")
        img = np.array(img, dtype=np.uint8)  # H,W,C in uint8

        # read first integer in the txt with same stem
        stem = os.path.splitext(os.path.basename(ip))[0]
        with open(os.path.join(self.label_dir, stem + ".txt"), "r") as f:
            label = int(f.read().strip().split()[0])

        # return (image as HWC uint8, label int)
        return img, label

def deepchecks_collate(samples) -> BatchOutputFormat:
    images = [s[0] for s in samples]      # list of HxWxC uint8 arrays
    labels = [int(s[1]) for s in samples] # list of ints
    return BatchOutputFormat(images=images, labels=labels)

def create_label_map(label_dir):
    label_set = set()
    for fname in os.listdir(label_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, fname), "r") as f:
            label_set.add(int(f.read().strip().split()[0]))
    return {i: f"class_{i}" for i in sorted(label_set)}

def test_data(report_dir, path_start, batch_size=64):
    train_image_path = os.path.join(path_start, 'data/raw/train/images')
    train_label_path = os.path.join(path_start, 'data/raw/train/labels')
    test_image_path  = os.path.join(path_start, 'data/raw/test/images')
    test_label_path  = os.path.join(path_start, 'data/raw/test/labels')

    train_ds_torch = PNGTxtDataset(train_image_path, train_label_path)
    test_ds_torch  = PNGTxtDataset(test_image_path,  test_label_path)

    train_loader = DataLoader(train_ds_torch, batch_size=batch_size, shuffle=True, collate_fn=deepchecks_collate)
    test_loader  = DataLoader(test_ds_torch,  batch_size=batch_size, shuffle=True, collate_fn=deepchecks_collate)

    label_map = create_label_map(train_label_path)

    # Wrap loaders with VisionData
    train_dc = VisionData(train_loader, task_type='classification', label_map=label_map)
    test_dc  = VisionData(test_loader,  task_type='classification', label_map=label_map)

    suite = train_test_validation()
    result = suite.run(train_dc, test_dc)

    os.makedirs(report_dir, exist_ok=True)
    assert result.passed, "Data validation failed. Check the report for details."
    result.save_as_html(os.path.join(path_start+"/"+report_dir, 'data_testing_report.html'))

# pip install "deepchecks[vision]" torch pillow matplotlib

from deepchecks.vision.suites import data_integrity
from deepchecks.vision import VisionData, BatchOutputFormat
from torch.utils.data import DataLoader

# ... keep your PNGTxtDataset, deepchecks_collate, and create_label_map as is ...

def test_data_quality(report_dir, path_start, batch_size=64):
    train_image_path = os.path.join(path_start, 'data/raw/train/images')
    train_label_path = os.path.join(path_start, 'data/raw/train/labels')

    train_ds_torch = PNGTxtDataset(train_image_path, train_label_path)
    train_loader   = DataLoader(train_ds_torch, batch_size=batch_size,
                                shuffle=True, collate_fn=deepchecks_collate)

    label_map = create_label_map(train_label_path)

    # Wrap single dataset (no test set needed) 
    train_dc = VisionData(train_loader, task_type='classification', label_map=label_map)

    suite = data_integrity()               
    result = suite.run(train_dc)           

    os.makedirs(report_dir, exist_ok=True)
    result.save_as_html(os.path.join(path_start, report_dir, 'data_quality_report.html'))


test_data(report_dir='reports/data', path_start='/Users/laiavillagrasa/Documents/UNI/TAED2/REPO/TAED2_SignAI')
test_data_quality(report_dir='reports/data', path_start='/Users/laiavillagrasa/Documents/UNI/TAED2/REPO/TAED2_SignAI')
