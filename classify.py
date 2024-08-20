import torch
import cv2
import numpy as np
from model import *
from torchvision import transforms
from dataloader import *
from dataloader2 import *

# Load the model weights

model = MVCNN()
model.load_state_dict(torch.load('./best_modeL.pt'))
# model.load_state_dict(torch.load('./test_model.pt'))
# model.load_state_dict(torch.load('test_model.pt'))
# print(model.state_dict())
# model = torch.load('./best_model.pt')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToTensor()
    ])

top_image_dir = './Data/top/grade_reject/r_010_t.jpg'
side_image_dir = './Data/side/grade_reject/r_010_s.jpg'

def preprocess_image(image):
    image = cv2.resize(image, (227, 227))

    # Normalize
    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()
    image = (image - min_val) / (max_val - min_val)

    # Transpose and convert to PyTorch tensor
    image = image.transpose((2, 0, 1))  # Change to (channels, height, width)
    image = torch.from_numpy(image)

    # Add a batch dimension
    image = image.unsqueeze(0)

    # Move to device (if using GPU)
    image = image.to(device)
    return image

model.eval()

data_dir = './Data'

test_data = GuavaDataset(data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False)

num_correct_a = 0
num_correct_b = 0
num_correct_c = 0
num_correct_r = 0
indeks = 0

with torch.no_grad():
    target = []
    prediksi = []
    for top_images, side_images, label in test_loader:
        top_images, side_images, label = top_images.to(device), side_images.to(device), label.to(device)
        outputs = model(top_images, side_images)

        # print(outputs)

        class_labels = ['grade_a', 'grade_b', 'grade_c', 'grade_reject']

        _, predictions = outputs.max(1)
        # print(predictions)
        predicted_class = class_labels[predictions]
        # print(predicted_class)
        if(predicted_class == 'grade_a' and label == 0.0):
            num_correct_a += 1
        elif(predicted_class == 'grade_b' and label == 1.0):
            num_correct_b += 1
        elif(predicted_class == 'grade_c' and label == 2.0):
            num_correct_c += 1
        elif(predicted_class == 'grade_reject' and label == 3.0):
            num_correct_r += 1

        indeks += 1

        # target.extend(labels.cpu().numpy())
        # prediksi.extend(predictions.cpu().numpy())

    # print(target)
    # print(prediksi)


print(f'Kelas A yang benar diprediksi: {num_correct_a} / 163')
print(f'Kelas B yang benar diprediksi: {num_correct_b} / 137')
print(f'Kelas C yang benar diprediksi: {num_correct_c} / 152')
print(f'Kelas Reject yang benar diprediksi: {num_correct_r} / 48')