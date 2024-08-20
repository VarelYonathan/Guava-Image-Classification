import torch
import cv2
import numpy as np
from model import *
from torchvision import transforms

# Load the model weights
model = MVCNN()
# model.load_state_dict(torch.load('./best_model.pt'))
model.load_state_dict(torch.load('C:/Users/ASUS/Desktop/Coding/Python/Skripsi/Final/MVCNN Jambu/best_modeL.pt'))

model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def preprocess_image(image):
    # Convert BGR (OpenCV format) to RGB (PyTorch format) if needed
    # if image.shape[2] == 3:  # Check if it already has 3 channels
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize (replace with your model's expected size)
    image = cv2.resize(image, (227, 227))  # Example for a 224x224 input

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

# top = cv2.VideoCapture('./Senin, 8 Mei 2023/Kamera Atas/Camera_B_1.mp4')
# side = cv2.VideoCapture('./Senin, 8 Mei 2023/Kamera Samping/Camera_A_1.mp4')
top = cv2.VideoCapture('./Senin, 8 Mei 2023/Kamera Atas/Camera_B_18.mp4')
side = cv2.VideoCapture('./Senin, 8 Mei 2023/Kamera Samping/Camera_A_18.mp4')

while True:
    ret1, frame1 = top.read()
    ret2, frame2 = side.read()

    if not ret1:
        break  # Exit the loop if video capture fails

    # Preprocess the frame (if needed)
    frame_processed1 = preprocess_image(frame1.copy())
    frame_processed2 = preprocess_image(frame2.copy())

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(frame_processed1, frame_processed2)

    # print(outputs)
    # print(torch.argmax(outputs, dim=1))
    class_labels = ['grade_a', 'grade_b', 'grade_c', 'grade_reject']

    predicted_class_index = torch.argmax(outputs, dim=1).item()
    predicted_class = class_labels[predicted_class_index]

    cv2.putText(frame1, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # cv2.putText(frame2, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video with MVCNN Model', frame1)
    # cv2.imshow('Video with MVCNN Model', frame2)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# # Release resources
top.release()
side.release()
cv2.destroyAllWindows()


