Summary
This project leverages a variety of technologies, from computer vision (OpenCV) to deep learning frameworks (TensorFlow/Keras), and emphasizes real-time processing, model training, and evaluation using Siamese Networks. The integration of GPU acceleration, evaluation metrics, and dataset preprocessing ensures a robust and scalable face recognition system.

1. End-to-End Workflow Description:

Image Capture:
Users can capture anchor, positive, or negative images by pressing keys (a, p, or n).

Preprocessing:
Captured images are resized to 100x100 pixels and normalized for consistency.
Images are labeled (anchor, positive, or negative) and stored in organized directories.

Training the Siamese Network:
Anchor-positive and anchor-negative pairs are created.
The Siamese Network is trained to minimize distances for matching pairs and maximize distances for non-matching pairs.
Training progress is monitored using metrics like loss, precision, and recall.

Real-Time Verification:
The user captures a live input image for verification.
The input image is compared against stored validation images.
If the similarity score exceeds the detection threshold, access is granted.

Access Control:
If access is granted, a green LED lights up, and the system sends a signal to unlock the door or barrier.
If access is denied, a red LED lights up.

Network Training

The training process of the Siamese Network involves the following steps:

Technologies Used:

1. Programming Language:
Python: The core language for implementing the system, chosen for its versatility and robust ecosystem for machine learning and computer vision.

2. Libraries and Frameworks:
Image Processing and Visualization

OpenCV:
Used for image and video capture, preprocessing (cropping, resizing), and real-time frame display.
Converts images to grayscale and applies basic transformations for feature enhancement.

Matplotlib:
Visualizes data, such as input images, validation images, and test results.
Provides a clear graphical representation of prediction outputs.

Deep Learning Framework

TensorFlow/Keras:
Powers the construction, training, and evaluation of deep learning models.

Functional API:
Utilized to define custom neural network architectures like the embedding model and Siamese Network.

Gradient Tape:
Tracks operations to compute gradients for training, allowing manual control over backpropagation.

Loss Functions:
Binary cross-entropy is used to measure prediction error during training.

Optimizers:
Adam optimizer is employed for adaptive learning rate optimization.

Data Handling

NumPy:
Handles numerical operations, including scaling, array manipulation, and tensor operations.
Supports image normalization (scaling pixel values between 0 and 1).

UUID (Universally Unique Identifier):
Generates unique file names for saving images, ensuring no overwrites during dataset creation.

3. Machine Learning Concepts

Siamese Network Architecture

Embedding Layer:
Extracts high-dimensional representations (feature vectors) from input images.
Uses convolutional layers, max pooling, and dense layers to create a feature-rich representation of input images.

Distance Layer (L1Dist):
Custom layer to calculate the absolute difference (L1 distance) between embeddings of anchor and validation images.
This enables comparison of feature vectors to measure similarity.

Classification Layer:
Outputs a single value (0 or 1) indicating whether two input images belong to the same class (match/no match).
Data Augmentation and Preprocessing

Image Preprocessing:
Resizes images to a uniform dimension (100x100) for consistent model input.
Normalizes pixel values to a range between 0 and 1 for stable model training.

Dataset Creation:
Combines anchor, positive, and negative image pairs into a dataset.
Labels pairs with 1 (positive pair) or 0 (negative pair).

4. Hardware Acceleration

GPU Acceleration:
TensorFlow GPU configuration is included to utilize available GPUs for faster training and inference.
Configures memory growth to avoid out-of-memory errors during large batch training.

5. Dataset Management

Dataset Paths:
Organizes images into folders: anchor, positive, and negative for training and evaluation.
Leverages TensorFlow's tf.data.Dataset API for efficient data loading and preprocessing.

6. Evaluation Metrics

Precision and Recall:
Metrics to evaluate the performance of the Siamese Network.
Precision measures the fraction of correctly identified positive pairs.
Recall measures the fraction of true positive pairs identified.

7. Checkpoints and Model Saving

Model Checkpoints:
Saves model weights during training for restoration and evaluation.
Uses TensorFlow's tf.train.Checkpoint for checkpointing.

Model Saving:
Saves the trained Siamese Network to an HDF5 file for future inference.

8. Real-Time Verification

Webcam Integration:
Captures real-time video frames for verification.
Allows live testing of the trained model by comparing captured frames with stored validation images.

Thresholds:
Detection Threshold: Determines the cutoff for similarity in predictions.
Verification Threshold: Calculates the proportion of positive matches to verify identity.

9. Loss Functions

Binary Cross-Entropy Loss:
Quantifies the difference between predicted and actual labels in binary classification tasks.
Essential for training the Siamese Network to distinguish between matched and unmatched pairs.

10. Model Architecture

Embedding Model:
Uses multiple convolutional layers to extract features from input images.
Final dense layer outputs a feature vector (embedding) of fixed size.

Siamese Network:
Combines two embedding models to process anchor and validation images.
Computes the similarity between embeddings and predicts whether the inputs are a match.

![image](https://github.com/user-attachments/assets/0761ee81-771d-4998-ada1-7a216a7ca5c1)

Complete Application

![image](https://github.com/user-attachments/assets/6bfa9aa2-c762-431a-97a3-a1669e4b9ead)

Block Diagram of the Access Control System with Facial Recognition

![image](https://github.com/user-attachments/assets/07198e2c-2bc1-4e2f-a3db-4683795bc586)
