import matplotlib.pyplot as plt


def plotting_evolution(history):
    # Loss history
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    # Accuracy history
    training_accuracy = history.history["accuracy"]
    validation_accuracy = history.history["val_accuracy"]

    # Figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot training & validation loss values
    ax1.plot(training_loss, label="Training Loss")
    ax1.plot(validation_loss, label="Validation Loss")
    ax1.set_title("Model Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # Plot training & validation accuracy values
    ax2.plot(training_accuracy, label="Training Accuracy")
    ax2.plot(validation_accuracy, label="Validation Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()
