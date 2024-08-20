import os
import torch
import cv2
import numpy as np
from torchvision import transforms

class GuavaDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.image_paths = []
    self.labels = []
    self.transform = transform

    view = os.listdir(data_dir)
    top_view_path = os.path.join(data_dir, view[0])
    side_view_path = os.path.join(data_dir, view[1])
    top_images = []
    side_images = []
    # self.global_min_val = None
    # self.global_max_val = None

    for class_name in os.listdir(top_view_path):
      top_class_path = os.path.join(top_view_path, class_name)
      for filename in os.listdir(top_class_path):
        top_image_path = os.path.join(top_class_path, filename)
        top_images.append(top_image_path)
        label = class_name
        if label == "grade_a":
          label = 0.0
        elif label == "grade_b":
          label = 1.0
        elif label == "grade_c":
          label = 2.0
        elif label == "grade_reject":
          label = 3.0
        self.labels.append(torch.tensor(label))

    for class_name in os.listdir(side_view_path):
      side_class_path = os.path.join(side_view_path, class_name)
      for filename in os.listdir(side_class_path):
        side_image_path = os.path.join(side_class_path, filename)
        side_images.append(side_image_path)

    n = 0
    # for img in top_images:
    for n in range(0,len(top_images), 1):
      self.image_paths.append([top_images[n], side_images[n]])
      # n += 1
    # self._calculate_global_min_max()

  def __len__(self):
    return len(self.image_paths)
  
  # def _calculate_global_min_max(self):
  #   for image_path in self.image_paths:
  #     image = cv2.imread(image_path[0])
  #     if self.global_min_val is None or image.min() < self.global_min_val:
  #       self.global_min_val = image.min()
  #     if self.global_max_val is None or image.max() > self.global_max_val:
  #       self.global_max_val = image.max()

  def __getitem__(self, idx):
    paired_image_path = self.image_paths[idx]
    top_image = cv2.imread(paired_image_path[0])
    side_image = cv2.imread(paired_image_path[1])
    if self.transform is not None:
      top_image = top_image.astype('float32')
      top_image = cv2.resize(top_image, (227, 227))
      side_image = side_image.astype('float32')
      side_image = cv2.resize(side_image, (227, 227))

      # Calculate min-max values for this image pair
      # top_min_val = np.amin(top_image, axis=(0, 1))
      # top_max_val = np.amax(top_image, axis=(0, 1))
      # side_min_val = np.amin(side_image, axis=(0, 1))
      # side_max_val = np.amax(side_image, axis=(0, 1))
      top_min_val = top_image.min()
      top_max_val = top_image.max()
      side_min_val = side_image.min()
      side_max_val = side_image.max()

      top_image = (top_image - top_min_val) / (top_max_val - top_min_val)
      side_image = (side_image - side_min_val) / (side_max_val - side_min_val)
      top_image = self.transform(top_image)
      side_image = self.transform(side_image)
    label = self.labels[idx]

    return top_image, side_image, label