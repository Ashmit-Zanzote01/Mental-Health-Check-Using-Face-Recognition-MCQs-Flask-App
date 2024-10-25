import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras_tuner import HyperModel, RandomSearch
import numpy as np
import matplotlib.pyplot as plt

# Set TensorFlow environment variable to suppress custom operation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Ensure proper UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')

# Data loading - Update paths to your actual dataset directories
train_data_dir = r"D:\Python\Health_AI\archive\train"
val_data_dir = r"D:\Python\Health_AI\archive\validation"  # Adjust path for validation data if available

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical'
)

# Class mapping
class_mapping = train_generator.class_indices
print("Class Indices Mapping:", class_mapping)

# Define the HyperModel for tuning
class CNNHyperModel(HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        # Input layer
        model.add(layers.InputLayer(input_shape=(48, 48, 3)))
        
        # Add convolutional layers with tunable hyperparameters
        model.add(layers.Conv2D(filters=hp.Int('conv1_units', 32, 128, step=32),
                                kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=hp.Int('conv2_units', 32, 192, step=32),
                                kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(filters=hp.Int('conv3_units', 32, 384, step=32),
                                kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'))
        model.add(layers.Dense(7, activation='softmax'))  # 7 classes

        # Compile the model with tunable optimizer and learning rate
        optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta'])
        learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')

        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)  # Using SGD as GD is not available
        elif optimizer_choice == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_choice == 'adagrad':
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        else:  # adadelta
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

# Define the tuner
tuner = RandomSearch(
    CNNHyperModel(),
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='hyperparam_search',
    project_name='emotion_detection'
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir='logs', histogram_freq=1)
checkpoint_callback = ModelCheckpoint(filepath='best_model.keras', save_best_only=True)

# Begin search
tuner.search(train_generator,
             epochs=25,
             validation_data=val_generator,
             callbacks=[early_stopping, tensorboard_callback, checkpoint_callback])  # Use early stopping and callbacks

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Optionally, save the best model
best_model.save('best_emotion_model.keras')

# Load the model to verify it
model = tf.keras.models.load_model('best_emotion_model.keras')
model.summary()  # Ensure the model is loaded correctly

# Extract and plot optimizer vs accuracy for each trial
results = tuner.oracle.get_best_trials(num_trials=20)
optimizers = [trial.hyperparameters.get('optimizer') for trial in results]

accuracies = [trial.score for trial in results]


# Plotting
plt.figure(figsize=(10, 5))
plt.plot(optimizers, accuracies, marker='o')
plt.title('Optimizer vs Accuracy')
plt.xlabel('Optimizer')
plt.ylabel('Validation Accuracy')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()
