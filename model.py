# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:18:40 2025

@author: ASUS

ISY503 Intelligent Systems

Assessment 3  

Students: 
        1. Aide Alejandra Navarro - student ID: A00152439
        2. Manuela Muñoz Ramirez - student ID:  A00140607
        3. Julieth Milena Sanchez Jimenez - student ID: A00157788
        4. Thiago Silva
        
Teacher: Husamuddin Mohammed

Date: 07/05/2025

=================================================
First File : Script to create and train the model
=================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Configuration parameters
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3  # NVIDIA model dimensions
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
STEERING_CORRECTION = 0.2  # Correction factor for left/right images

# Define custom loss function that will be correctly saved with the model
def mse(y_true, y_pred):
    """Mean Squared Error loss function"""
    return tf.reduce_mean(tf.square(y_pred - y_true))

def load_data(data_dir):
    """
    Load image paths and assign steering angles based on directory
    """
    images = []
    steering_angles = []
    
    # Define steering values for each directory
    steering_map = {
        'Left': 0.7,      # Left steering
        'Right': -0.7,    # Right steering
        'Forward': 0.0    # Center steering (straight)
    }
    
    # Load each directory
    for direction, base_angle in steering_map.items():
        directory = os.path.join(data_dir, direction)
        if os.path.exists(directory):
            # Get all image files in the directory
            for img_file in os.listdir(directory):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(directory, img_file)
                    images.append(img_path)
                    steering_angles.append(base_angle)
    
    # Convert to numpy arrays for consistency
    return np.array(images), np.array(steering_angles)


def preprocess_image(img_path):
    """
    Preprocess image for the neural network:
    - Load image
    - Crop to remove irrelevant parts (sky, car hood)
    - Resize to NVIDIA model dimensions
    - Convert to YUV color space (as in NVIDIA paper)
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    # Convert from BGR to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get dimensions
    h, w = img.shape[:2]
    
    # Crop the image (remove sky and car hood)
    top_crop = min(int(h * 0.35), h - 1)  # Remove top portion
    bottom_crop = min(int(h * 0.1), h - top_crop - 1)  # Remove bottom portion
    
    if h > (top_crop + bottom_crop):
        img = img[top_crop:h-bottom_crop, :, :]
    
    # Resize to NVIDIA model dimensions
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    
    # Convert to YUV color space (as in NVIDIA paper)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    return img


def augment_image(img, steering_angle):
    """
    Apply data augmentation to increase dataset variety:
    - Random horizontal flip with steering angle adjustment
    - Random brightness adjustment
    - Random shadow addition
    """
    # Random horizontal flip with steering angle inversion
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        steering_angle = -steering_angle
    
    # Random brightness adjustment
    if random.random() > 0.5:
        # Convert to HSV for easier brightness manipulation
        hsv = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)  # First to RGB
        hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2HSV)
        
        # Adjust brightness
        ratio = 0.4 + random.random() * 0.6  # 0.4 to 1.0
        hsv[:,:,2] = hsv[:,:,2] * ratio
        
        # Convert back to YUV
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    
    # Random shadow (creates horizontal or vertical shadow)
    if random.random() > 0.7:
        h, w = img.shape[:2]
        is_horizontal = random.random() > 0.5
        
        if is_horizontal:
            shadow_height = int(random.uniform(0.3, 0.7) * h)
            shadow_width = w
            x1, x2 = 0, w
            y1 = y2 = shadow_height
        else:
            shadow_height = h
            shadow_width = int(random.uniform(0.3, 0.7) * w)
            y1, y2 = 0, h
            x1 = x2 = shadow_width
            
        mask = np.zeros_like(img[:,:,0])
        shadow_dimension = np.random.uniform(0.4, 0.6)  # Random shadow intensity
        
        # Create a polygon for the shadow region
        vertices = np.array([[(x1, y1), (x2, y2), (w, h), (0, h)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 1)
        
        # Add the shadow
        for c in range(3):
            img[:,:,c] = img[:,:,c] * (1 - mask * shadow_dimension)
    
    return img, steering_angle


def batch_generator(images, steering_angles, batch_size=32, is_training=True):
    """
    Generate batches of training/validation data
    """
    num_samples = len(images)
    
    while True:  # Loop forever (needed for Keras fit_generator)
        # Shuffle data for each epoch
        indices = np.random.permutation(num_samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            
            X_batch = []
            y_batch = []
            
            for i in batch_indices:
                img_path = images[i]
                angle = steering_angles[i]
                
                try:
                    # Load and preprocess image
                    img = preprocess_image(img_path)
                    
                    # Apply augmentation during training
                    if is_training:
                        img, angle = augment_image(img, angle)
                    
                    X_batch.append(img)
                    y_batch.append(angle)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            # Skip this batch if empty
            if len(X_batch) == 0:
                continue
                
            # Convert to numpy arrays
            yield np.array(X_batch), np.array(y_batch)


def create_nvidia_model():
    """
    Create the NVIDIA end-to-end self-driving model architecture with regularization
    """
    model = Sequential([
        # Normalization layer
        Lambda(lambda x: x/127.5 - 1.0, input_shape=INPUT_SHAPE),
        
        # Convolutional layers with ELU activation (better than ReLU for self-driving)
        Conv2D(24, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        
        # Flatten before fully connected layers
        Flatten(),
        
        # Fully connected layers with dropout for regularization
        Dense(100, activation='elu'),
        Dropout(0.5),
        Dense(50, activation='elu'),
        Dropout(0.2),
        Dense(10, activation='elu'),
        
        # Output layer (single continuous value for steering angle)
        Dense(1)
    ])
    
    # Compile model with MSE loss and Adam optimizer
    # Using our custom mse function that will be correctly serialized with the model
    model.compile(loss=mse, optimizer=Adam(learning_rate=LEARNING_RATE))
    
    return model


def train_model(model, X_train, y_train, X_valid, y_valid):
    """
    Train the model with data generators and callbacks
    """
    # Create data generators
    train_generator = batch_generator(X_train, y_train, batch_size=BATCH_SIZE, is_training=True)
    valid_generator = batch_generator(X_valid, y_valid, batch_size=BATCH_SIZE, is_training=False)
    
    # Calculate steps per epoch
    steps_per_epoch = max(len(X_train) // BATCH_SIZE, 1)
    validation_steps = max(len(X_valid) // BATCH_SIZE, 1)
    
    # Model checkpoint to save best model
    checkpoint = ModelCheckpoint(
        'model_checkpoint.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    return history


def plot_training_history(history):
    """
    Plot training and validation loss
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.savefig('training_history.png')
    plt.close()
    
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")


def main():
    # Path to dataset
    data_dir = './training_data'  # Update if your path is different
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    print("Loading images and steering data...")
    
    # Load data
    X, y = load_data(data_dir)
    
    # Make sure we have data
    if len(X) == 0:
        print("Error: No images found!")
        return
        
    print(f"Dataset loaded: {len(X)} images")
    
    # Split data into training and validation sets (80/20)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_valid)} samples")
    
    # Create model
    model = create_nvidia_model()
    model.summary()
    
    # Train the model
    history = train_model(model, X_train, y_train, X_valid, y_valid)
    
    # Plot training results
    plot_training_history(history)
    
    # Save final model
    model.save('model.h5', save_format='h5')
    print("Model saved as 'model.h5'")
    
    # Additional information for the user
    print("\nModel training complete!")
    print("To use this model with the simulator, run:")
    print("python drive.py model.h5")


if __name__ == "__main__":
    main()