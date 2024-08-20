import torch
from torch import nn, optim
from torchvision import transforms
from model import *
from dataloader import *
from train_model import *
from evaluate import *
from plot import *

# path
top_train_dir = "./train/top"
side_train_dir = "./train/side"
top_test_dir = "./test/top"
side_test_dir = "./test/side"

train_dir = "./train/"
test_dir = "./test/"

transform = transforms.Compose([
    transforms.ToTensor()
    ])

# Datasets dan dataloaders
train_data = GuavaDataset(train_dir, transform=transform)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)


test_data = GuavaDataset(test_dir, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

# model, optimizer, dan loss function
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MVCNN()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
# learning_rate = 0.00001
# learning_rate = 0.00005
# learning_rate = 0.000075
learning_rate = 0.0001
# learning_rate = 0.00025
# learning_rate = 0.0005
# learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
# num_epochs = 500
# num_epochs = 200
# num_epochs = 150
# num_epochs = 100
num_epochs = 75
# num_epochs = 50
# num_epochs = 25

# PATH = "./model_test.pt"
# PATH = "./model.pt"
# PATH = "./model_1000.pt"
# PATH = "./model_4_cnn.pt"
# PATH = "./model_3_cnn.pt"
# PATH = "./model_2_cnn.pt"
# PATH = "./model_1_cnn.pt"

# PATH = "./best_modeL.pt"
# PATH = "./test_modeL.pt"
PATH = "./best_model_candidate.pt"

result, best_model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, 50)
torch.save(best_model.state_dict(), PATH)

# evaluation(model, test_loader, criterion, optimizer, device)
evaluation(best_model, test_loader, device)

# Call the function to plot
plot_metrics(result)