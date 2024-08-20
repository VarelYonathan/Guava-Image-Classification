import torch
from sklearn.metrics import classification_report

def evaluation(model, val_loader, device):
  model.eval()
  target = []
  prediksi = []

  with torch.no_grad():

    for top_images, side_images, labels in val_loader:
      top_images, side_images, labels = top_images.to(device), side_images.to(device), labels.to(device)

      outputs = model(top_images, side_images)

      print(outputs)

      _, predictions = outputs.max(1)

      target.extend(labels.cpu().numpy())
      prediksi.extend(predictions.cpu().numpy())

  print(classification_report(target, prediksi, zero_division=1))

def evaluation_atas(model, val_loader, criterion, optimizer, device):
  model.eval()
  target = []
  prediksi = []

  with torch.no_grad():

    for top_images, side_images, labels in val_loader:
      top_images, labels = top_images.to(device), labels.to(device)

      outputs = model(top_images)

      _, predictions = outputs.max(1)

      target.extend(labels.cpu().numpy())
      prediksi.extend(predictions.cpu().numpy())

  print(classification_report(target, prediksi, zero_division=1))

def evaluation_samping(model, val_loader, criterion, optimizer, device):
  model.eval()
  target = []
  prediksi = []

  with torch.no_grad():

    for top_images, side_images, labels in val_loader:
      side_images, labels = side_images.to(device), labels.to(device)

      outputs = model(side_images)

      _, predictions = outputs.max(1)

      target.extend(labels.cpu().numpy())
      prediksi.extend(predictions.cpu().numpy())

  print(classification_report(target, prediksi, zero_division=1))