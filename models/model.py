import os
import datetime
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from data_loader import DataLoader  # Assuming your custom data loader
from tensorflow.keras import layers, models # type: ignore

# Define constants (ensure these paths are correct)
BASE = ''
models_path = os.path.join(BASE, 'models')
datasets_path = os.path.join(BASE, 'dataset')

train_file_name = f"{datasets_path}/train.csv"
test_file_name = f"{datasets_path}/test.csv"

# Check GPU availability
print(f"Number of GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")
device = "/GPU:0" if tf.config.experimental.list_physical_devices("GPU") else "/CPU:0"
print("Device:", device)

# Model Building Function
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Softmax for binary classification
    ])
    return model


if __name__ == "__main__":
    # Initialize DataLoader
    dataloader = DataLoader(train_csv=train_file_name, test_csv=test_file_name, train_dir=datasets_path, test_dir=datasets_path)
    dataloader.load_data()

    # Get train, validation, and test datasets
    train_data = dataloader.get_train_data()  # Image and label dataset
    val_data = dataloader.get_val_data()     # Image and label dataset
    test_data = dataloader.get_test_data()   # Test images without labels

    # Set number of classes for binary classification
    num_classes = 2  
    input_shape = (128, 128, 3)
    
    # Create and compile the model
    model = create_cnn_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Setup TensorBoard callback
    run_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + run_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    start_time = time.time()  # Record start time

    # Train the model
    with tf.device(device):
        history = model.fit(train_data, epochs=10, validation_data=val_data, verbose=1, callbacks=[tensorboard_callback])
    
    end_time = time.time()  # Record end time
    training_time = end_time - start_time  # Calculate training time

    # Create a folder for this run using the current timestamp or a run number
    run_folder = os.path.join(models_path, f"runs/{run_time}")
    os.makedirs(run_folder, exist_ok=True)

    # Save the model to the run folder
    model_save_path = os.path.join(run_folder, 'model.h5')
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Predict model on test data
    print("Predicting on test data")
    predictions = model.predict(test_data, verbose=2)

    # Convert predictions to class labels (if binary classification, we assume it's softmax output)
    predicted_labels = np.argmax(predictions, axis=1)

    # Store predictions in a CSV file
    ids = np.arange(1, len(dataloader.test_df) + 1)
    results_df = pd.DataFrame({
        'id': ids,
        'label': predicted_labels
    })

    # Save predictions to CSV file in the run folder
    predictions_file = os.path.join(run_folder, 'predictions.csv')
    results_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    # Instructions to run TensorBoard
    print(f"\nTotal training time: {training_time:.2f} seconds")
    print("\nTo visualize TensorBoard, run the following command in your terminal:")
    print(f"tensorboard --logdir={log_dir}")
