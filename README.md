# CPU Burst Time Prediction Using Machine Learning

This project explores machine learning (ML)-based methods for predicting CPU burst times of processes to enhance scheduling algorithms like Shortest Job First (SJF) and Shortest Remaining Time First (SRTF).  
Traditional methods like Exponential Averaging (EA) offer simple predictions but often lack accuracy for dynamic workloads.  
Our goal is to improve prediction quality using real-world workload attributes, focusing on both non-historical and historical features.

---

## Dataset

- **Source:** GWA-T-4 AuverGrid dataset.
- **Jobs Selected:** First 5000 entries (after cleaning, 4340 jobs used).
- **Features Used:**  
  - Non-Historical: SubmitTime, UserID, Used Memory, ReqTime, ReqMemory  
  - Historical: WaitTime, NProcs, AverageCPUTimeUsed, Status, etc.
  
---

## Machine Learning Models

- **K-Nearest Neighbors (KNN)** – $k=5$
- **Decision Tree (DT)**
- **Random Forest (RF)** – 100 estimators
- **Multi-Layer Perceptron (MLP)** – 4 hidden layers, ReLU activation
- **Exponential Averaging (EA)** – Baseline method

---

## Files Description

| File | Description |
|-----|-------------|
| `OS_Project_ML_models.ipynb` | Training and evaluation of ML models (historical attributes) |
| `OS_Project_MLP.ipynb` | Specific code for MLP training and evaluation |
| `non_hist_OS_Project_ML_models.ipynb` | ML training with non-historical attributes |
| `non_hist_OS_Project_MLP.ipynb` | MLP with non-historical attributes |
| `mlp_burst_predictor.pth` | Trained MLP model (historical features) |
| `non_hist_mlp_burst_predictor.pth` | Trained MLP model (non-historical features) |
| `processes_datasets.csv` | Cleaned dataset used for training/testing |
| `historical_features_exact_reasons.csv` | Features with explanations (historical) |
| `non_historical_features_exact_reasons.csv` | Features with explanations (non-historical) |
| `Model_Performance_Summary.csv` | Performance metrics for ML models (historical) |
| `non_hist_Model_Performance_Summary.csv` | Performance metrics (non-historical) |
| `inference_time_summary.csv` | Inference time measurements for each model |

---

## Results Summary

- **ML models outperform Exponential Averaging** in all evaluated metrics: MAE, R², CC, and RAE.
- **Best Performance:** Random Forest (historical features).
- **Fastest Inference:** Decision Tree (almost as fast as Exponential Averaging).
- **Non-Historical Features:** Still yielded reasonably accurate predictions, supporting early job scheduling decisions.

---

## How to Run

1. Clone the repository.
2. Install required Python packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib torch
3. Open the .ipynb notebooks and run them sequentially to train models and evaluate performance.

## References
- "A machine learning-based approach to estimate the CPU-burst time for processes in the computational grids."
2015 3rd International Conference on Artificial Intelligence, Modelling and Simulation (AIMS). IEEE, 2015.
