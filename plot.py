import matplotlib.pyplot as plt

def plot_metrics(log_dict):
    epochs = range(1, len(log_dict['training_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, log_dict['training_loss'], 'r', label='Training loss')
    plt.plot(epochs, log_dict['validation_loss'], 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, log_dict['training_accuracy'], 'r', label='Training accuracy')
    plt.plot(epochs, log_dict['validation_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()