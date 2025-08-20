# EEG-Based Neurological Disorder Classification
## Rice Datathon 2025 - Neurotech Track 🥈


**🏆 2nd Place Winner - Rice Datathon 2025 Neurotech@Rice Track**

A machine learning approach to classify neurological disorders using EEG (Electroencephalography) data. This project implements multiple classification algorithms to identify various neurological conditions from brain wave patterns.

##  Competition Overview

This project was developed for the Rice Datathon 2025 Neurotech Track, focusing on the automated diagnosis of neurological disorders through EEG signal analysis.

### Key Objectives
- Classify 6 types of neurological disorders plus healthy controls
- Process and analyze high-dimensional EEG data
- Implement robust machine learning pipelines
- Achieve high accuracy while handling class imbalance

##  Dataset

The dataset contains EEG recordings with the following characteristics:
- **Training samples**: ~1,500 subjects
- **Features**: 120+ EEG channel measurements across multiple frequency bands
- **Target classes**: 7 categories (6 disorders + healthy control)
  - Addictive disorder
  - Anxiety disorder
  - Mood disorder
  - Obsessive compulsive disorder
  - Schizophrenia
  - Trauma and stress related disorder
  - Healthy control

### Data Structure
- `Train_and_Validate_EEG.csv`: Training and validation data
- `Test_Set_EEG.csv`: Test data for final predictions
- Features include absolute band powers (AB) across different frequency bands (theta, alpha, beta, gamma, delta)

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
Git
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rice-datathon-2025.git
cd rice-datathon-2025
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place data files in the `data/` directory:
```
data/
├── Train_and_Validate_EEG.csv
└── Test_Set_EEG.csv
```

### Running the Analysis

#### For Model Training and Evaluation:
```bash
jupyter notebook notebooks/eeg_disorder_classification.ipynb
```
Keep `PRODUCTION_MODE = False` in the configuration cell.

#### For Competition Predictions:
1. Set `PRODUCTION_MODE = True` in the configuration cell
2. Run all cells in the notebook
3. Find predictions in `results/xgb_predictions.csv`

#### For Correlation Analysis:
```bash
python visualizations/correlation_analysis.py
```

## 📁 Project Structure

```
RiceDatathon2025/
│
├── notebooks/
│   └── eeg_disorder_classification.ipynb  # Main analysis notebook
│
├── visualizations/
│   └── correlation_analysis.py           # EEG correlation analysis
│
├── data/                                 # Data directory (not in repo)
│   ├── Train_and_Validate_EEG.csv
│   └── Test_Set_EEG.csv
│
├── results/                              # Output directory
│   ├── xgb_predictions.csv              # XGBoost predictions
│   ├── svm_predictions.csv              # SVM predictions
│   ├── rf_predictions.csv               # Random Forest predictions
│   └── visualizations/                  # Analysis plots
│
├── docs/                                 # Documentation
├── requirements.txt                      # Python dependencies
├── README.md                            # This file
```

##  Models Implemented

### 1. XGBoost (Primary Model)
- **Best performing model** with highest accuracy
- Gradient boosting with optimized hyperparameters
- Handles class imbalance effectively
- Feature importance analysis included

### 2. Support Vector Machine (SVM)
- RBF kernel with balanced class weights
- Robust to outliers
- Good for high-dimensional data

### 3. Random Forest
- Ensemble method with 100 estimators
- Balanced class weights for handling imbalance
- Feature importance ranking

### 4. Logistic Regression (Binary Classification)
- One-vs-Rest approach for each disorder
- L2 regularization
- Useful for understanding individual disorder patterns

##  Data Processing Pipeline

1. **Data Cleaning**
   - Remove unnecessary columns (ID, date, specific disorder)
   - Handle missing values (mean imputation for test data)
   
2. **Feature Engineering**
   - Remove highly correlated features (>0.95 correlation)
   - Encode categorical variables
   - Standardize all features

3. **Dimensionality Reduction**
   - Correlation-based feature selection
   - Removal of redundant EEG channels

##  Key Findings

- **Feature Importance**: Certain EEG frequency bands show stronger predictive power
- **Class Performance**: Some disorders (e.g., Schizophrenia) are easier to identify than others
- **Correlation Patterns**: High correlation within frequency bands across electrodes

##  Model Performance

| Model | Validation Accuracy | Key Strengths |
|-------|-------------------|---------------|
| XGBoost | ~75% | Best overall performance, feature importance |
| SVM | ~70% | Robust to outliers |
| Random Forest | ~72% | Good ensemble performance |
| Logistic Regression | 69-91% (binary) | Excellent for specific disorders |

##  Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebook

##  References

1. Park, S. M. (2021, August 16). EEG machine learning. Retrieved from [osf.io/8bsv](https://osf.io/8bsv)
2. [Scikit-learn Documentation](https://scikit-learn.org/)
3. [XGBoost Documentation](https://xgboost.readthedocs.io/)
4. [Gradient Boosting Decision Trees Introduction](https://www.machinelearningplus.com/machine-learning/an-introduction-to-gradient-boosting-decision-trees/)

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- Rice Datathon 2025 organizers
- Neurotech@Rice for providing the dataset
---


