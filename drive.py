"""
==================================================================
Second File: Script to connect the trained model to the simulator
==================================================================

Self-driving car control script for the Udacity simulator
Implements lane detection, steering control, and recovery strategies

"""

"""
Usage: python drive.py model.h5
"""


import argparse
import base64
from io import BytesIO
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque

# Configure socketio server
sio = socketio.Server(async_mode='eventlet', cors_allowed_origins=['*'])
app = Flask(__name__)
model = None

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
MAX_STEERING_ANGLE = 15.0 #Maximum allowed steering angle in degrees
STEERING_MULTIPLIER = 0.85  # we reduce from 1.2 to 0.85 for stability
# Subtle left bias to compensate for right-side tendency
CENTER_BIAS = -0.02  

# Buffer for steering angle smoothing
STEERING_BUFFER_SIZE = 5
steering_angles = deque(maxlen=STEERING_BUFFER_SIZE)

# Define the custom loss function for model loading
def mse(y_true, y_pred):
    """Mean Squared Error loss function"""
    return tf.reduce_mean(tf.square(y_pred - y_true))

def preprocess_image(image):
    """
   Preprocess image for the neural network:
   - Crop to remove irrelevant parts (sky, car hood)
   - Resize to model input dimensions
   - Convert to YUV color space (as in NVIDIA paper)
   """
    # Get dimensions
    h, w = image.shape[:2]
    
    # Crop image - focusing on the road
    top_crop = int(h * 0.35)
    bottom_crop = int(h * 0.1)
    
    if h > (top_crop + bottom_crop):
        cropped = image[top_crop:h-bottom_crop, :, :]
    else:
        cropped = image
    
    # Resize to model input dimensions
    resized = cv2.resize(cropped, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    
    # Convert to YUV color space (as used in NVIDIA paper)
    yuv = cv2.cvtColor(resized, cv2.COLOR_RGB2YUV)
    
    return yuv

def detect_road_type(image):
    """
    Road type detection
    """
    h, w = image.shape[:2]
    
    # Use the bottom portion of the image where the road is most visible
    bottom_portion = image[int(h*0.6):h, :]
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(bottom_portion, cv2.COLOR_RGB2HSV)
    
    # Extract average saturation and value
    avg_sat = np.mean(hsv[:,:,1])
    avg_val = np.mean(hsv[:,:,2])
    
    # Simple classification
    if avg_sat < 70 and 40 < avg_val < 180:
        return "asphalt"
    elif avg_sat > 20 and avg_val > 100:
        return "desert"
    else:
        return "unknown"

def detect_lane_boundaries(image):
    """
    lane boundary detection with proper type handling
    """
    h, w = image.shape[:2]
    
    # Use the bottom third of the image where the road is most visible
    bottom_portion = image[int(h*0.6):h, :]
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(bottom_portion, cv2.COLOR_RGB2HSV)
    
    # Detect road type
    road_type = detect_road_type(image)
    
    # Create masks for different road types - with explicit type conversion
    if road_type == "asphalt":
        # Asphalt road (gray/black) - explicitly use np.uint8 for all bounds
        lower_gray = np.array([0, 0, 40], dtype=np.uint8)
        upper_gray = np.array([180, 40, 160], dtype=np.uint8)
        road_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    elif road_type == "desert":
        # Desert/dirt (tan/brown) - explicitly use np.uint8 for all bounds
        lower_tan = np.array([10, 25, 90], dtype=np.uint8)
        upper_tan = np.array([35, 160, 230], dtype=np.uint8)
        road_mask = cv2.inRange(hsv, lower_tan, upper_tan)
    else:
        # Fallback - try to detect both types with proper type handling
        lower_gray = np.array([0, 0, 40], dtype=np.uint8)
        upper_gray = np.array([180, 40, 160], dtype=np.uint8)
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        lower_tan = np.array([10, 25, 90], dtype=np.uint8)
        upper_tan = np.array([35, 160, 230], dtype=np.uint8)
        tan_mask = cv2.inRange(hsv, lower_tan, upper_tan)
        
        road_mask = cv2.bitwise_or(gray_mask, tan_mask)
    
    # Morphological operations to clean up noise
    kernel = np.ones((5,5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    
    # Divide image into left, center, right regions for analysis
    h_mask, w_mask = road_mask.shape
    left = np.sum(road_mask[:, :w_mask//3])
    center = np.sum(road_mask[:, w_mask//3:2*w_mask//3])
    right = np.sum(road_mask[:, 2*w_mask//3:])
    
    # Calculate road position metrics
    total = max(left + center + right, 1)  # Avoid division by zero
    left_ratio = left / total
    right_ratio = right / total
    
    # Calculate confidence in detection
    confidence = min(1.0, total / (h_mask * w_mask * 255 * 0.3))
    
    # Start with default bias
    correction = CENTER_BIAS
    
    # Very simple correction strategy
    if left_ratio < 0.2:  # Road imbalanced to right
        correction = -0.03  # Gentle right correction
    elif right_ratio < 0.2:  # Road imbalanced to left
        correction = 0.03  # Gentle left correction
    
    return correction, confidence, road_type

def apply_smoothing(steering_angle, steering_angles_queue):
    
    # Add current angle to buffer
    steering_angles_queue.append(steering_angle)
    
    # Return early if not enough data points
    if len(steering_angles_queue) <= 1:
        return steering_angle
    
    # Calculate simple weighted average (more weight to recent angles)
    weights = [0.5, 0.3, 0.15, 0.03, 0.02][:len(steering_angles_queue)]
    weights.reverse()  # Newest values get higher weights
    weights = np.array(weights) / sum(weights)
    
    smoothed_angle = 0
    for i, angle in enumerate(steering_angles_queue):
        smoothed_angle += angle * weights[i]
    
    return smoothed_angle

# Track state
last_steering = 0.0  # Track previous steering angle

@sio.on('telemetry')
def telemetry(sid, data):
    global steering_angles, last_steering
    
    if data:
        try:
            # Get current data
            current_speed = float(data["speed"])
            
            # Decode image
            image = Image.open(BytesIO(base64.b64decode(data["image"])))
            image_array = np.asarray(image)
            
            # Process image for model
            processed_image = preprocess_image(image_array)
            
            # Predict steering angle from neural network
            raw_steering = float(model.predict(np.array([processed_image]), verbose=0)[0][0])
            
            # Apply steering multiplier
            model_steering = raw_steering * STEERING_MULTIPLIER
            
            # Get road type for debug info
            road_type = detect_road_type(image_array)
            
            # Get basic lane correction
            lane_correction, confidence, detected_road_type = detect_lane_boundaries(image_array)
            
            # Add correction to model steering
            steering_angle = model_steering + lane_correction
            
            # Apply minimal smoothing
            steering_angle = apply_smoothing(steering_angle, steering_angles)
            
            # Limit change rate for stability
            max_change = 0.1
            if abs(steering_angle - last_steering) > max_change:
                if steering_angle > last_steering:
                    steering_angle = last_steering + max_change
                else:
                    steering_angle = last_steering - max_change
            
            # Very conservative throttle based on steering angle
            if abs(steering_angle) > 0.15:
                throttle = 0.06  # Very slow for sharp turns
            elif abs(steering_angle) > 0.1:
                throttle = 0.07  # Slow for moderate turns
            else:
                # Always conservative throttle
                throttle = 0.08
                
            # Lower throttle even more in desert
            if road_type == "desert":
                throttle = max(0.06, throttle * 0.8)
            
            # Safety cap on steering
            if abs(steering_angle) > 0.6:
                steering_angle = 0.6 if steering_angle > 0 else -0.6
            
            # Save current steering for next iteration
            last_steering = steering_angle
            
            # Print debug info
            print(f"Speed: {current_speed:.1f}, Raw: {raw_steering:.4f}, " +
                  f"Final: {steering_angle:.4f}, Throttle: {throttle:.2f}" +
                  f", Road: {road_type}")
            
            # Send control commands to simulator
            send_control(steering_angle, throttle)
            
        except Exception as e:
            print(f"Error in telemetry: {e}")
            import traceback
            traceback.print_exc()
    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("Connected", sid)
    send_control(0, 0.07)  # Start with neutral steering and conservative throttle

@sio.on('disconnect')
def disconnect(sid):
    print("Disconnected", sid)

def send_control(steering_angle, throttle):
    """Send control commands to the simulator"""
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-Driving Car Simulator Control')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=0.08,
        help='Base throttle setting for straight roads'
    )
    parser.add_argument(
        '--multiplier',
        type=float,
        default=0.85,
        help='Steering multiplier to adjust responsiveness'
    )
    args = parser.parse_args()
    
    # Update settings from command line if provided
    if args.multiplier != 0.85:
        STEERING_MULTIPLIER = args.multiplier
        print(f"Using custom steering multiplier: {STEERING_MULTIPLIER}")
    
    # Load model
    try:
        # Ensure memory growth for GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load model with custom objects dictionary
        custom_objects = {'mse': mse}
        model = load_model(args.model, custom_objects=custom_objects)
        print(f"Model {args.model} loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Wrap Flask application with socketio middleware
    app = socketio.Middleware(sio, app)
    
    # Start server
    print("Starting server at port 4567")
    try:
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except OSError as e:
        if "address already in use" in str(e).lower():
            print("Port 4567 is already in use. Make sure no other instances are running.")
            print("You may need to restart your computer or kill processes using this port.")
        else:
            print(f"Server error: {e}")