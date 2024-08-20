import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.drop1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        # self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.relu6 = nn.ReLU()
        # self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # self.relu7 = nn.ReLU()

        # self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # x = self.drop1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # x = self.drop2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)

        # x = self.pool3(x)

        # x = self.conv6(x)
        # x = self.relu6(x)

        # x = self.conv7(x)
        # x = self.relu7(x)

        # x = self.pool4(x)
        return x
    
class MVCNN(nn.Module):
  def __init__(self):
    super(MVCNN, self).__init__()
    self.top = CNN()
    self.side = CNN()
    # Perhitungan jumlah elemen yang dihasilkan oleh flatten
    input_top = torch.randn(1, 3, 227, 227)
    input_side = torch.randn(1, 3, 227, 227)
    output_top = self.top(input_top)
    output_side = self.side(input_side)
    concat = torch.cat((output_top, output_side), dim=1)
    flatten = nn.Flatten()(concat)
    flatten_size = flatten.size(1)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(flatten_size, 64)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(64, 64)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(64, 4)

  def forward(self, input_t, input_s):
    vector_output_t = self.top(input_t)
    vector_output_s = self.side(input_s)
    concat = torch.cat((vector_output_t, vector_output_s), dim=1)
    x = self.flatten(concat)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x
  
class CNN_Single(nn.Module):
  def __init__(self):
    super(CNN_Single, self).__init__()
    self.view = CNN()
    # Perhitungan jumlah elemen yang dihasilkan oleh flatten
    input = torch.randn(1, 3, 227, 227)
    output = self.view(input)
    flatten = nn.Flatten()(output)
    flatten_size = flatten.size(1)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(flatten_size, 64)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(64, 64)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(64, 4)

  def forward(self, input):
    vector_output = self.view(input)
    x = self.flatten(vector_output)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x