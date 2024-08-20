import torch
from torch import nn, optim
from torchvision import transforms
from model import *
from dataloader import *
from train_model_single_view import *
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

test_data = GuavaDataset(test_dir, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# model, optimizer, dan loss function
torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_single = CNN_Single()
model_single = model_single.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = optim.Adam(model_single.parameters(), lr=learning_rate)
num_epochs = 500

# result_atas, best_model = train_model_view_atas(model_single, train_loader, test_loader, criterion, optimizer, num_epochs, device, 50)
result_samping, best_model = train_model_view_samping(model_single, train_loader, test_loader, criterion, optimizer, num_epochs, device, 50)

# PATH = "./model_view_atas.pt"
PATH = "./model_view_samping.pt"
torch.save(best_model.state_dict(), PATH)

# target, prediksi = evaluation(model_single, test_loader, criterion, optimizer, device)
# print(classification_report(target, prediksi, zero_division=1))
# evaluation_atas(model_single, test_loader, criterion, optimizer, device)
# evaluation_samping(model_single, test_loader, criterion, optimizer, device)

# evaluation_atas(best_model, test_loader, criterion, optimizer, device)
evaluation_samping(best_model, test_loader, criterion, optimizer, device)

# Call the function to plot
# plot_metrics(result_atas)
plot_metrics(result_samping)