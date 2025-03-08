# Libraries Used

This project utilizes several libraries for various purposes, including machine learning, image processing, and web development. Below is a list of the libraries used and their respective purposes:

## Flask
- **Version:** >=2.0.0
- **Purpose:** Flask is a lightweight WSGI web application framework. It is used to create the web server and handle HTTP requests and responses.
- **Why it's necessary:** Flask provides the foundation for our web application, allowing us to create routes, render templates, and handle user interactions. It is essential for serving the web interface where users can upload images and view inference results.

## Ultralytics
- **Version:** >=8.0.20
- **Purpose:** Ultralytics provides the YOLO (You Only Look Once) object detection model. It is used for detecting objects in images and video feeds.
- **Why it's necessary:** YOLO is a state-of-the-art object detection model that enables real-time detection of objects in images and videos. It is crucial for identifying and localizing different types of plastics in the input data.

## NumPy
- **Version:** >=1.22.0
- **Purpose:** NumPy is a library for numerical computing in Python. It is used for array operations and mathematical functions.
- **Why it's necessary:** NumPy provides efficient array operations and mathematical functions that are essential for preprocessing images and handling numerical data during model inference.

## Torch
- **Version:** >=1.11.0
- **Purpose:** Torch is a machine learning library for Python, used for building and training neural networks. It is used in conjunction with YOLO for object detection.
- **Why it's necessary:** Torch provides the core functionalities for building and training neural networks, which are essential for the YOLO model to perform object detection.

## Torchvision
- **Version:** >=0.12.0
- **Purpose:** Torchvision is a library that provides datasets, model architectures, and image transformations for computer vision. It is used alongside Torch.
- **Why it's necessary:** Torchvision offers utilities for working with image data, including transformations and pre-trained models, which are essential for processing images and integrating with the YOLO model.

## OpenCV
- **Version:** >=4.5.5
- **Purpose:** OpenCV (Open Source Computer Vision Library) is used for image and video processing. It is used to read, process, and display images and video frames.
- **Why it's necessary:** OpenCV provides a wide range of image and video processing functions, such as reading images, resizing, and drawing bounding boxes. It is essential for handling input data and visualizing inference results.

## Kaggle
- **Purpose:** Kaggle is an online community and platform for data science and machine learning. It is used for accessing datasets, running experiments, and collaborating with other data scientists.
- **Why it's necessary:** Kaggle provides a rich repository of datasets and a collaborative environment that helped us in training our model. It offers powerful tools and resources for data exploration, model training, and evaluation, which were crucial for the success of our project.

## Matplotlib
- **Version:** >=3.2.2
- **Purpose:** Matplotlib is a plotting library for Python. It is used to create visualizations, such as displaying cropped images with labels.
- **Why it's necessary:** Matplotlib allows us to create visualizations of the inference results, such as displaying cropped images with their predicted labels. This is important for presenting the results in a user-friendly manner.

## Pandas
- **Version:** >=1.3.0
- **Purpose:** Pandas is a data manipulation and analysis library. It is used for handling and processing data in tabular form.
- **Why it's necessary:** Pandas provides powerful data structures and functions for manipulating and analyzing data, which are useful for managing and processing the results of the inference.

## PyYAML
- **Version:** >=5.3.1
- **Purpose:** PyYAML is a YAML parser and emitter for Python. It is used for reading and writing YAML configuration files.
- **Why it's necessary:** PyYAML allows us to read and write configuration files in YAML format, which is useful for managing model configurations and other settings in a human-readable format.

## Seaborn
- **Version:** >=0.11.0
- **Purpose:** Seaborn is a statistical data visualization library based on Matplotlib. It is used for creating attractive and informative statistical graphics.
- **Why it's necessary:** Seaborn provides high-level functions for creating attractive and informative statistical graphics, which are useful for visualizing data distributions and model performance metrics.

## TQDM
- **Version:** >=4.64.0
- **Purpose:** TQDM is a library for creating progress bars. It is used to display progress bars for loops and other iterative processes.
- **Why it's necessary:** TQDM provides progress bars that help monitor the progress of long-running tasks, such as model training and inference, making it easier to track the status of these processes.

## TensorBoard
- **Version:** >=2.4.1
- **Purpose:** TensorBoard is a visualization toolkit for TensorFlow. It is used for visualizing training metrics and model performance.
- **Why it's necessary:** TensorBoard allows us to visualize training metrics and model performance, which is essential for monitoring and debugging the training process of the CNN model.

## Requests
- **Version:** >=2.23.0
- **Purpose:** Requests is a simple HTTP library for Python. It is used for making HTTP requests.
- **Why it's necessary:** Requests provides a simple and efficient way to make HTTP requests, which can be useful for interacting with external APIs and services.

## TensorFlow
- **Version:** >=2.6.0
- **Purpose:** TensorFlow is an open-source machine learning library. It is used for building and training the CNN (Convolutional Neural Network) model for plastic classification.
- **Why it's necessary:** TensorFlow provides the tools and functionalities for building and training the CNN model, which is essential for classifying different types of plastics based on the detected objects.

---
Feel free to reach out if you have any questions or need further clarification on the libraries used in this project.
