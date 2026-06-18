# Brain Hemorrhage CT Image Segmentation and Classification

## Project Overview
An intracranial hemorrhage is a life-threatening medical condition where bleeding occurs within the skull. Prompt and accurate detection via CT imaging is crucial for deciding on surgical interventions or medications. 

This project aims to automate the detection, classification, and segmentation of intracranial hemorrhages from CT scans using various Machine Learning and Deep Learning architectures. We explored Logistic Regression, Support Vector Machines (SVM), Convolutional Neural Networks (CNNs), and a U-Net semantic segmentation model to assist in localizing and measuring brain bleeding.

## Dataset
The dataset was provided by **Zeta Surgical**, a biotechnology company based in Boston.
* **Volume:** 752,794 layered (post-sliced) CT scans of brain hemorrhages.
* **Format:** $512 \times 512$ JPG images.
* **Features:** Four images per hemorrhage taken at different angles and "windows" (e.g., brain window for tissue granularity, bone window for skull features).
* **Labels:** Coordinate-based polygons marking the exact location of the hemorrhage, provided via `.csv` files for different hemorrhage subtypes:
  1. Subdural
  2. Epidural
  3. Subarachnoid
  4. Intraparenchymal
  5. Intraventricular
  6. Multiple

## Project Structure
The repository consists of several Jupyter Notebooks dedicated to data preprocessing, mask generation, and model training:

* **`RemoveFlaggedEntries.ipynb`**: Data cleaning pipeline. Reads a `flagged.txt` file containing IDs of corrupted or mislabeled scans and actively removes them from the respective hemorrhage `.csv` label files.
* **`GenerateMasks.ipynb`**: Uses `OpenCV` (`cv2.fillPoly` and `cv2.drawContours`) to parse the polygon coordinates from the `.csv` files and generate $512 \times 512$ binary image masks representing the ground-truth hemorrhage areas.
* **`LinearRegression.ipynb`**: Contains baseline statistical models, including Logistic Regression and SVMs (via `scikit-learn`), paired with cross-validation and standard scaling techniques.
* **`ConvolutionalNeuralNetwork.ipynb`**: Contains the deep learning classification pipeline built with TensorFlow/Keras. Includes data loading, model architecture definitions (Conv2D, MaxPooling, Dropout), and performance visualization.
* **`MATH4570_Final_Project-1.pdf`**: The comprehensive final research paper detailing the methodology, architecture, and evaluation metrics of the project.

## Methodology & Models

### 1. Data Preprocessing
* **Corrupt Data Removal:** Flagged/unreadable images were stripped from the training pipeline.
* **Mask Generation:** Transformed coordinate strings into binary NumPy arrays/images to serve as ground truth for segmentation.
* **Normalization & Resizing:** For computationally heavy baseline models, images were downscaled (e.g., to $100 \times 100$) and normalized.

### 2. Logistic Regression
* **Setup:** Binary and multi-class prediction based on flattened image arrays.
* **Results:** Achieved an accuracy of ~87% after scaling to 1000 iterations. While effective, it served primarily as a baseline and exhibited the lowest comparative accuracy.

### 3. Support Vector Machines (SVM)
* **Setup:** Implemented using a Radial Basis Function (RBF) kernel to transform data into higher-dimensional spaces for easier separation.
* **Optimization:** Hyperparameters were tuned using Grid Search, utilizing extra features like labeler agreement and labeling state.

### 4. Convolutional Neural Network (CNN)
* **Architecture:** 2x `Conv2D` layers (with 32, 64, 128 filters), 2x `MaxPooling2D` layers, followed by a `Flatten` layer and `Dense` layers with `Dropout` (0.5) to prevent overfitting.
* **Results:** Trained over 10 epochs using categorical cross-entropy and the Adam optimizer. The model generalized well, achieving a final validation accuracy of **~81.88%** on multi-class classification.

### 5. U-Net Image Segmentation
* **Architecture:** An encoder-decoder Convolutional Neural Network designed specifically for biomedical image segmentation (11 layers total: 5 contraction, 5 expansion).
* **Implementation:** To combat heavy class imbalance (where healthy tissue heavily outnumbers hemorrhage pixels), data classes were weighted at a 7:3 ratio. 
* **Results:** Trained on 2,804 inputs over 50 epochs. Achieved a **Jaccard Coefficient of 99.3%** and an IOU of ~15%. Future work includes transitioning from JPG to DICOM formats to improve bounding box centralization and mask accuracy.

## Installation & Requirements
To run the notebooks in this repository, you will need the following Python libraries:
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `opencv-python` (`cv2`)
* `scikit-learn`
* `tensorflow` / `keras`
* `ast` (Standard library)

## 📝 Acknowledgements
Special thanks to Zeta Surgical for providing the extensive CT scan dataset and labeling coordinates that made this research possible.
