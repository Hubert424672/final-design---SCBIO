# Import necessary libraries
import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report
import cv2

# Define base paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # Path to current script
source_dir = os.path.join(base_dir, "animal")          # Folder with original data
target_base = os.path.join(base_dir, "animal_split")   # Folder to store train/val split
train_dir = os.path.join(target_base, "train")         # Train folder
val_dir = os.path.join(target_base, "val")             # Validation folder
split_ratio = 0.8                                       # 80% for training, 20% for validation
img_size = (224, 224)                                   # Image dimensions for model input
batch_size = 32                                         # Batch size for training

# Split dataset if not already done
if not os.path.exists(train_dir):
    print(" Creating train/val split...")
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    for cls in os.listdir(source_dir):  # Loop over each class folder
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        for phase, img_list in zip(["train", "val"], [train_imgs, val_imgs]):
            dest_dir = os.path.join(target_base, phase, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(cls_path, img), os.path.join(dest_dir, img))

# Data generators for model input
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load data from folders into generators
train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical")
val_data = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False)

# Build the model using Functional API
inputs = tf.keras.Input(shape=(224, 224, 3))  # Input layer
base_model = MobileNetV2(include_top=False, input_tensor=inputs, weights="imagenet")  # Load pretrained MobileNetV2
base_model.trainable = False  # Freeze the base model

x = base_model.output                                # Get base model output
x = layers.GlobalAveragePooling2D()(x)               # Apply global average pooling
x = layers.Dense(128, activation='relu')(x)          # Fully connected layer
outputs = layers.Dense(train_data.num_classes, activation='softmax')(x)  # Output layer

model = tf.keras.Model(inputs, outputs)              # Final model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model

# Train the model
model.fit(train_data, validation_data=val_data, epochs=5)

# Evaluate the model
val_data.reset()
y_pred = model.predict(val_data)                     # Get predictions
y_true = val_data.classes                            # True labels
y_pred_classes = np.argmax(y_pred, axis=1)           # Predicted class indices
class_names = list(val_data.class_indices.keys())    # List of class names

# Print classification report
print("\n Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Function to generate Class Activation Map (CAM)
def generate_cam_image(image_path, class_index, model, base_model, last_conv_layer_name='Conv_1'):
    original_img = cv2.imread(image_path)                        # Load original image (for clearer background)
    original_img = cv2.resize(original_img, img_size)            # Resize image to match model input

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)  # Load image for preprocessing
    img_array = tf.keras.preprocessing.image.img_to_array(img)                     # Convert to array
    img_pre = preprocess_input(np.expand_dims(img_array, axis=0))                 # Preprocess
    img_tensor = tf.convert_to_tensor(img_pre)                                    # Convert to tensor

    conv_layer = base_model.get_layer(last_conv_layer_name)       # Get last convolutional layer
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[conv_layer.output, model.output])  # Grad-CAM model

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)        # Forward pass
        loss = predictions[:, class_index]                        # Focus on specific class

    grads = tape.gradient(loss, conv_outputs)[0]                  # Get gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))             # Average gradients over spatial dimensions
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)  # Compute CAM heatmap

    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)     # Normalize heatmap
    heatmap = cv2.resize(heatmap.numpy(), img_size)               # Resize to match original image
    heatmap = np.uint8(255 * heatmap)                             # Convert to 0-255 range
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)        # Apply color map

    cam = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)     # Overlay heatmap on original image
    return cam

# Generate CAMs for each class
cam_images = []
titles = []
for cls in class_names:
    cls_dir = os.path.join(val_dir, cls)
    img_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if img_list:
        img_path = os.path.join(cls_dir, img_list[0])             # Use one example image per class
        cam_img = generate_cam_image(img_path, class_names.index(cls), model, base_model)
        cam_images.append(cam_img)
        titles.append(cls)

# Display CAMs in a grid
cols = 5
rows = int(np.ceil(len(cam_images) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(20, 8))

for i in range(rows * cols):
    ax = axs[i // cols, i % cols] if rows > 1 else axs[i]
    if i < len(cam_images):
        ax.imshow(cv2.cvtColor(cam_images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        ax.set_title(titles[i])
    ax.axis('off')

plt.tight_layout()
output_path = os.path.join(base_dir, "all_cams.png")  # Save CAM visualization
plt.savefig(output_path)
plt.show()
print(f"\n CAM saved as: {output_path}")
