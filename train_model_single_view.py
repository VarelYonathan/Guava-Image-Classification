import torch
# from dataloader import *
# from model import *
import time

class EarlyStopping:
  def __init__(self, min_delta=0., patience=0):
      self.min_delta = min_delta
      self.patience = patience
      self.best_val = None
      self.wait = 0

  def __call__(self, val_loss):
      current = val_loss
      if self.best_val is None or current < self.best_val - self.min_delta:
          self.best_val = current
          self.wait = 0
          return False
      else:
          self.wait += 1
          if self.wait >= self.patience:
              return True
          return False

def train_model_view_atas(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=0, min_delta=0.):
  early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
  log = {
        'training_loss': [],
        'validation_loss': [],
        'training_accuracy': [],
        'validation_accuracy': []
    }
  # start, end = 0, 0
  start = time.time()
  best_train_loss = 99
  best_train_loss_epoch = 0
  best_val_loss = 99
  best_val_loss_epoch = 0
  best_train_accuracy = 0
  best_train_accuracy_epoch = 0
  best_val_accuracy = 0
  best_val_accuracy_epoch = 0

  best_model = None
  wait = 0

  for epoch in range(num_epochs):
    model.train()
    print(f'Epoch: {epoch+1}/{num_epochs}')
    train_loss = 0.0
    num_correct = 0

    for top_images, side_images, labels in train_loader:
      labels = labels.type(torch.LongTensor)
      top_images, labels = top_images.to(device), labels.to(device)

      # Clear gradients from previous iteration
      optimizer.zero_grad()

      # Forward pass
      outputs = model(top_images)
      loss = criterion(outputs, labels)

      predictions = torch.argmax(outputs, dim=1)

      _, predictions = outputs.max(1)
      num_correct += predictions.eq(labels).sum().item()

      # Backward pass dan update weights
      loss.backward()
      optimizer.step()

      # Update running loss
      train_loss += loss.item()

    # Print epoch loss statistics
    epoch_loss = train_loss / len(train_loader)
    train_accuracy = num_correct / len(train_loader.dataset)
    log['training_accuracy'].append(train_accuracy)
    log['training_loss'].append(epoch_loss)
    if(epoch_loss < best_train_loss):
      best_train_loss = epoch_loss
      best_train_loss_epoch = epoch+1
    if(train_accuracy > best_train_accuracy):
      best_train_accuracy = train_accuracy
      best_train_accuracy_epoch = epoch+1
    print(f'Training Loss: {epoch_loss:.3f}, Training Accuracy: {train_accuracy:.3f}')


    model.eval()

    with torch.no_grad():  # Disable perhitungan gradient untuk evaluasi
      eval_loss = 0.0
      num_correct = 0

      for top_images, side_images, labels in val_loader:
        labels = labels.type(torch.LongTensor)
        top_images, labels = top_images.to(device), labels.to(device)

        outputs = model(top_images)
        loss = criterion(outputs, labels)

        eval_loss += loss.item()

        _, predictions = outputs.max(1)
        num_correct += predictions.eq(labels).sum().item()

      eval_epoch_loss = eval_loss / len(val_loader)
      val_accuracy = num_correct / len(val_loader.dataset)

      log['validation_accuracy'].append(val_accuracy)
      log['validation_loss'].append(eval_epoch_loss)
      print(f'Validation Loss: {eval_epoch_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')
      if(eval_epoch_loss < best_val_loss):
        best_val_loss = eval_epoch_loss
        best_model = model
        best_val_loss_epoch = epoch+1
        wait = 0
      else:
        wait += 1
        if wait >= patience:
          print("Early stopping triggered. Stopping training.")
          break
      if(val_accuracy > best_val_accuracy):
        best_val_accuracy = val_accuracy
        best_val_accuracy_epoch = epoch+1
      
  end = time.time()
  print(f'Waktu eksekusi: {(end-start)} detik')
  print(f'Best train loss: {(best_train_loss)} epoch :{(best_train_loss_epoch)}')
  print(f'Best train acc: {(best_train_accuracy)} epoch :{(best_train_accuracy_epoch)}')
  print(f'Best val loss: {(best_val_loss)} epoch :{(best_val_loss_epoch)}')
  print(f'Best val acc: {(best_val_accuracy)} epoch :{(best_val_accuracy_epoch)}')
  return log, best_model



def train_model_view_samping(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=0):
  log = {
        'training_loss': [],
        'validation_loss': [],
        'training_accuracy': [],
        'validation_accuracy': []
    }
  # start, end = 0, 0
  start = time.time()
  best_train_loss = 99
  best_train_loss_epoch = 0
  best_val_loss = 99
  best_val_loss_epoch = 0
  best_train_accuracy = 0
  best_train_accuracy_epoch = 0
  best_val_accuracy = 0
  best_val_accuracy_epoch = 0
  
  best_model = None
  wait = 0
  for epoch in range(num_epochs):
    model.train()
    print(f'Epoch: {epoch+1}/{num_epochs}')
    train_loss = 0.0
    num_correct = 0

    for top_images, side_images, labels in train_loader:
      labels = labels.type(torch.LongTensor)
      side_images, labels = side_images.to(device), labels.to(device)

      # Clear gradients from previous iteration
      optimizer.zero_grad()

      # Forward pass
      outputs = model(side_images)
      loss = criterion(outputs, labels)

      predictions = torch.argmax(outputs, dim=1)

      _, predictions = outputs.max(1)
      num_correct += predictions.eq(labels).sum().item()

      # Backward pass dan update weights
      loss.backward()
      optimizer.step()

      # Update running loss
      train_loss += loss.item()

    # Print epoch loss statistics
    epoch_loss = train_loss / len(train_loader)
    train_accuracy = num_correct / len(train_loader.dataset)
    log['training_accuracy'].append(train_accuracy)
    log['training_loss'].append(epoch_loss)
    if(epoch_loss < best_train_loss):
      best_train_loss = epoch_loss
      best_train_loss_epoch = epoch+1
    if(train_accuracy > best_train_accuracy):
      best_train_accuracy = train_accuracy
      best_train_accuracy_epoch = epoch+1
    print(f'Training Loss: {epoch_loss:.3f}, Training Accuracy: {train_accuracy:.3f}')

    model.eval()

    with torch.no_grad():  # Disable perhitungan gradient untuk evaluasi
      eval_loss = 0.0
      num_correct = 0

      for top_images, side_images, labels in val_loader:
        labels = labels.type(torch.LongTensor)
        side_images, labels = side_images.to(device), labels.to(device)

        outputs = model(side_images)
        loss = criterion(outputs, labels)

        eval_loss += loss.item()

        _, predictions = outputs.max(1)
        num_correct += predictions.eq(labels).sum().item()

      eval_epoch_loss = eval_loss / len(val_loader)
      val_accuracy = num_correct / len(val_loader.dataset)

      log['validation_accuracy'].append(val_accuracy)
      log['validation_loss'].append(eval_epoch_loss)
      print(f'Validation Loss: {eval_epoch_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')
      if(eval_epoch_loss < best_val_loss):
        best_val_loss = eval_epoch_loss
        best_model = model
        best_val_loss_epoch = epoch+1
        wait = 0
      else:
        wait += 1
        if wait >= patience:
          print("Early stopping triggered. Stopping training.")
          break
      if(val_accuracy > best_val_accuracy):
        best_val_accuracy = val_accuracy
        best_val_accuracy_epoch = epoch+1
      
  end = time.time()
  print(f'Waktu eksekusi: {(end-start)} detik')
  print(f'Best train loss: {(best_train_loss)} epoch :{(best_train_loss_epoch)}')
  print(f'Best train acc: {(best_train_accuracy)} epoch :{(best_train_accuracy_epoch)}')
  print(f'Best val loss: {(best_val_loss)} epoch :{(best_val_loss_epoch)}')
  print(f'Best val acc: {(best_val_accuracy)} epoch :{(best_val_accuracy_epoch)}')
  return log, best_model

