

# Real-time 2D Object Recognition

This project focuses on developing a real-time system for 2D object recognition. The system aims to recognize specific objects placed on a white surface from a top-down camera view, achieving translation, scale, and rotation invariance.

## Features

- **Object Segmentation**: The system employs thresholding, morphological filtering, and connected component analysis to segment individual objects from the input video frames.
- **Feature Extraction**: Various object features are extracted, including the axis of least central moment, oriented bounding box, aspect ratio, and other relevant features for classification.
- **Training Data Collection**: A dedicated training mode allows users to collect labeled feature vectors for different objects, facilitating model training.
- **Object Classification**: The system utilizes a nearest-neighbor approach with a scaled Euclidean distance metric to classify new objects based on their feature vectors.
- **Unknown Object Detection**: An extension is implemented to detect unknown objects not present in the object database.
- **Performance Evaluation**: The system's performance is evaluated using a confusion matrix, providing insights into classification accuracy and potential misclassifications.
- **Deep Learning Classification**: A pre-trained deep network is employed to generate embedding vectors for objects, followed by nearest-neighbor matching using cosine distance as the distance metric.

## Getting Started

To run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/real-time-2d-object-recognition.git`
2. Install the required dependencies (e.g., OpenCV, NumPy, scikit-learn, etc.).
3. Run the main script: `python main.py`

Refer to the project documentation for detailed instructions on configuration, training, and usage.

## Usage

1. Place the objects of interest on a white surface within the camera's field of view.
2. Run the main script to start the real-time object recognition system.
3. Use the designated key press (e.g., "O") to collect training data for new objects.
4. The system will display the recognized objects with their labels and other relevant information on the video output.

## To see the output check the link in the Report pdf

## Contributing

Contributions to the project are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements

We would like to express our gratitude to the following platforms for their invaluable support and resources throughout the project:

- Springer
- ResearchGate
- GitHub
- Stack Overflow
- ChatGPT
- Gemini
