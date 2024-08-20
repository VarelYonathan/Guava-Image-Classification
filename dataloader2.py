import os
import torch
import cv2
import numpy as np
from torchvision import transforms

class GuavaDataset2(torch.utils.data.Dataset):
  def __init__(self, top_image_path, side_image_path, transform=None):
    self.transform = transform

    self.top_image_path = top_image_path
    self.side_image_path = side_image_path

  def __len__(self):
    return 1

  def __getitem__(self, idx):
    top_image = cv2.imread(self.top_image_path)
    side_image = cv2.imread(self.side_image_path)
    if self.transform is not None:
      top_image = top_image.astype('float32')
      top_image = cv2.resize(top_image, (227, 227))
      side_image = side_image.astype('float32')
      side_image = cv2.resize(side_image, (227, 227))

      top_min_val = top_image.min()
      top_max_val = top_image.max()
      side_min_val = side_image.min()
      side_max_val = side_image.max()

      top_image = (top_image - top_min_val) / (top_max_val - top_min_val)
      side_image = (side_image - side_min_val) / (side_max_val - side_min_val)
      top_image = self.transform(top_image)
      side_image = self.transform(side_image)

    return top_image, side_image