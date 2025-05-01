# Predictive Maintenance using Machine Learning
This project applies machine learning to enable predictive maintenance in industrial settings, aiming to identify potential equipment failures before they occur. By analyzing datasets such as the Industrial Internet of Things (IIoT) dataset and a manual inspection dataset, the project employs unsupervised learning techniques to detect anomalies that may signal impending failures.

## Key Features
- **Anomaly Detection**: Identify irregularities in industrial equipment data to predict failures.
- **Machine Learning Algorithms**:
    - **K-Means Clustering**: Groups data into clusters and flags anomalies based on their distance from cluster centroids.
    - **Isolation Forest**: Detects outliers by isolating anomalies through recursive data partitioning.
    - **Autoencoders**: Reconstructs input data and identifies anomalies with high reconstruction errors.
    - **Gaussian Mixture Model (GMM)**: Models data as a combination of Gaussian distributions and detects anomalies using probability thresholds.
    - **DBSCAN**: A density-based clustering algorithm that identifies clusters and outliers based on data density.

- **Model Evaluation Metrics**:
    - Precision
    - Recall
    - F1-Score
    - Cohen's Kappa

## Datasets
- **IIoT Dataset**: Real-time sensor data from industrial machines (`iiot_30min_norm.csv`).
- **Manual Inspection Dataset**: Human-recorded inspection data, which may include errors (`manual_30min_norm.csv`).

## How to Run
1. Clone the repository:
     ```bash
     git clone https://github.com/qazisallar/predictive_maintenance.git
     ```
2. Install the required dependencies:
     ```bash
     pip install -r requirements.txt
     ```
3. Load the datasets in the Jupyter notebook:
     ```python
     iiot_data = pd.read_csv('data/iiot_30min_norm.csv')
     manual_data = pd.read_csv('data/manual_30min_norm.csv')
     ```
4. Execute the notebook to train and evaluate the models.

## Results
The models were assessed using metrics such as Precision, Recall, F1-Score, and Cohen's Kappa. Isolation Forest and DBSCAN demonstrated superior performance on the IIoT dataset, while K-Means and Autoencoders delivered consistent results across both datasets.
