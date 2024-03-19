import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from Augmentation import balance_dataset
import matplotlib.pyplot as plt

def load_data(directory):
    image_dims = (224, 224)

    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2 
    )

    train_generator = datagen.flow_from_directory(
        directory,
        target_size=image_dims,
        batch_size=32,
        class_mode='categorical',  
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        directory,
        target_size=image_dims,
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    class_labels = train_generator.class_indices
    num_classes = len(class_labels)

    return train_generator, validation_generator, num_classes

def build_model(base_model, num_classes):
    
    base_model.trainable = False
    
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(num_classes)

    model = tf.keras.Sequential([
      base_model,
      global_average_layer,
      prediction_layer
    ])

    base_learning_rate = 0.0001
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def plot_accuracy_loss(acc, val_acc, loss, val_loss):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')   

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def train_model(train_generator, validation_generator, epochs):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = build_model(base_model, num_classes)
    
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model
    loss, accuracy = model.evaluate(validation_generator)
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']   

    loss = history.history['loss']
    val_loss = history.history['val_loss']  

    plot_accuracy_loss(acc, val_acc, loss, val_loss)


if __name__ == "__main__":

    # balance_dataset('./datasets/images')
    directory = "./datasets/augmented_directory/images/Apple"
    
    train_generator, validation_generator, num_classes = load_data(directory)

    train_model(train_generator, validation_generator, epochs=7)


    

