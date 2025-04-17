import cv2
import pandas as pd
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['img_id'])
        label = self.df.iloc[idx]['text']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 64))
        img = self.transform(img)
        return img, label
