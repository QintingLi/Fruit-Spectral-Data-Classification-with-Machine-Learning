# Fruit Classification Project

This project implements various machine learning models to classify fruits based on reflectance and intensity data. The project compares different classification approaches including SVM, KNN, 1D CNN, and ResNet architectures.

## Project Structure

```
├── Intensity_Dataset/      # Directory containing intensity measurement data
├── Reflectance_Dataset/   # Directory containing reflectance measurement data
└── src/                   # Source code directory
```

## Features

- Data preprocessing and fusion from multiple sources
- Implementation of multiple classification models:
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - 1D Convolutional Neural Network (CNN)
  - 1D ResNet
  - Dual-channel CNN with Attention Mechanism

## Dependencies

- Python 3.x
- TensorFlow 2.x
- PyTorch
- scikit-learn
- pandas
- numpy
- imblearn
- seaborn
- matplotlib

## Data Processing

### Data Loading
- Supports multiple file encodings (UTF-8, GBK, Latin1)
- Handles missing values and data cleaning
- Combines intensity and reflectance datasets

### Preprocessing Steps
1. Data cleaning and normalization
2. Feature scaling using StandardScaler
3. Label encoding for classification
4. Train-test split (80-20)
5. SMOTE oversampling for imbalanced classes

## Model Architectures

### 1D CNN with Attention
- Dual-channel architecture for processing both intensity and reflectance data
- Custom attention mechanism for feature importance weighting
- Multiple convolutional layers with batch normalization
- Global average pooling and dense layers

### ResNet Architecture
- Basic residual blocks with skip connections
- Multiple stacked layers for deep feature extraction
- Batch normalization and ReLU activation
- Global average pooling for final feature aggregation

## Model Performance

The project includes comprehensive evaluation metrics:
- Classification accuracy
- Detailed classification reports
- Confusion matrices
- Training history visualization

## Usage

1. Data Preparation:
```python
# Load and preprocess data
data, labels = load_dataset_with_labels(DATASET_PATH)
X = preprocess_data(data)
```

2. Model Training:
```python
# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

3. Evaluation:
```python
# Evaluate model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

## Model Comparison Results

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| SVM   | 0.98     | -        |
| KNN   | 0.98     | -        |
| 1D CNN| 1.0      | -        |
| ResNet| 1.0      | -        |

## Future Improvements

1. Hyperparameter optimization
2. Model ensemble techniques
3. Cross-validation implementation
4. Additional data augmentation methods
5. Extended attention mechanism variations

