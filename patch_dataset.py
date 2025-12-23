from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    """
    Stand-alone dataset so DataLoader workers can import it.

    Args
    ----
    df      : pandas.DataFrame with columns path, label, wsi_id, x, y
    tfms    : torchvision transform pipeline
    """
    def __init__(self, df, tfms):
        self.df   = df.reset_index(drop=True)
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(row.path).convert("RGB")
        img   = self.tfms(img)
        label = torch.tensor(row.label, dtype=torch.float32)  # BCEWithLogits
        meta  = (row.wsi_id, row.x, row.y)
        return img, label, meta
