# Pottery Classification using Neural Network

## Project Overview
Multiclass classification of pottery types using a neural network on archaeological ceramic data from the Digital Archaeological Record (tDAR).

## Dependencies
- Python 3.10.12
- TensorFlow
- NumPy
- Pandas

## Dataset
- Source: Digital Archaeological Record (tDAR)
- Dataset: Ceramics: Temporal-Spatial Dataset (1988)
- tDAR ID: 6039
- DOI: 10.6067/XCV8TD9WNB

## Model Architecture
- Multilayer Perceptron (MLP)
- Input Layer: 95 neurons
- Hidden Layer: 64 neurons (ReLU activation)
- Output Layer: 9 neurons (Softmax activation)

## Key Techniques
- He Weight Initialization
- L2 Regularization
- Early Stopping
- Learning Rate Decay
- Adam Optimizer

## Performance
- Test Accuracy: 99.31%
- Test Loss: 0.0376

## Repository Structure
- `notebooks/`: Directory containing Jupyter notebooks for EDA and Modeling 
- `01_EDA.ipynb`: Data exploration and encoding notebook
- `02_Modeling.ipynb`: Main notebook with model architecture, training, and performance analysis 
- `data/`: Directory containing dataset - raw files, and numpy arrays ready for analysis
- `models/`: Trained pottery_classifier.v1.keras model

## How to Run
1. Clone the repository
2. Install required dependencies from requirements.txt
3. Open desired jupyter notebook
4. Run cells sequentially

## Citation
Digital Archaeological Record. (1988). Ceramics: Temporal-Spatial Dataset (tDAR id: 6039). https://doi.org/10.6067/XCV8TD9WNB

## License
MIT