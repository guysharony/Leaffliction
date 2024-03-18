import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from Augmentation import balance_dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer

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

    # Get class labels and number of classes
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

def train_model(model, train_generator, validation_generator, epochs):
    # Train the model
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

    # Evaluate the model
    loss, accuracy = model.evaluate(validation_generator)
    print("Validation Loss:", loss)
    print("Validation Accuracy:", accuracy)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']   

    loss = history.history['loss']
    val_loss = history.history['val_loss']  

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


    # base_model.trainable = True
    # print("Number of layers in the base model: ", len(base_model.layers))

    # # Fine-tune from this layer onwards
    # fine_tune_at = 100

    # # Freeze all the layers before the `fine_tune_at` layer
    # for layer in base_model.layers[:fine_tune_at]:
    #     layer.trainable =  False
    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    #           optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001/10),
    #           metrics=['accuracy'])
    # fine_tune_epochs = 2
    # total_epochs =  epochs + fine_tune_epochs

    # history_fine = model.fit(train_generator,
    #                          epochs=total_epochs,
    #                          initial_epoch =  history.epoch[-1],
    #                          validation_data=validation_generator)
    # acc += history_fine.history['accuracy']
    # val_acc += history_fine.history['val_accuracy']

    # loss += history_fine.history['loss']
    # val_loss += history_fine.history['val_loss']

    # plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.ylim([0.8, 1])
    # plt.plot([epochs-1,epochs-1],
    #         plt.ylim(), label='Start Fine Tuning')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')

    # plt.subplot(2, 1, 2)
    # plt.plot(loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.ylim([0, 1.0])
    # plt.plot([epochs-1,epochs-1],
    #         plt.ylim(), label='Start Fine Tuning')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('epoch')
    # plt.show()    
if __name__ == "__main__":
    
    # balance_dataset('./datasets/images')
    directory = "./datasets/augmented_directory/images/Apple"
    
    train_generator, validation_generator, num_classes = load_data(directory)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    model = build_model(base_model, num_classes)

    train_model(model, train_generator, validation_generator, epochs=4)
    

