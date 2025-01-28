# Supply Chain Multi-Output Prediction Model

This repository contains a machine learning solution for predicting multiple supply chain metrics using various regression models. The implementation includes Random Forest, XGBoost, and LightGBM models for multi-output prediction.

## Project Overview

The project focuses on predicting multiple supply chain metrics:
- LBL (Label)
- MTLp2 through MTLp16 (Multiple Target Labels)

## Model Architecture

### 1. Random Forest Model
- Base Model: RandomForestRegressor
- Configuration:
  - n_estimators: 100
  - max_features: sqrt
  - min_samples_split: 2
  - min_samples_leaf: 1
- Advantages:
  - Handles non-linear relationships
  - Robust to outliers
  - Provides feature importance
  - Less prone to overfitting

### 2. XGBoost Model
- Base Model: XGBRegressor
- Configuration:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8
- Advantages:
  - Gradient boosting for improved accuracy
  - Built-in regularization
  - Handles missing values efficiently
  - Optimized for speed and performance

### 3. LightGBM Model
- Base Model: LGBMRegressor
- Configuration:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
  - colsample_bytree: 0.8
- Advantages:
  - Faster training speed
  - Lower memory usage
  - Better handling of large datasets
  - Leaf-wise tree growth

### Multi-Output Architecture
- Implementation: MultiOutputRegressor
- Standardized preprocessing using StandardScaler
- Independent model training for each target
- Automated model comparison and selection

## Model Performance

### Overall Performance Comparison
1. XGBoost: Best overall performance (R² = 0.7684)
2. LightGBM: Close second (R² = 0.7650)
3. Random Forest: Good baseline (R² = 0.7463)

### Performance by Target Groups

#### Strong Predictions (R² > 0.80)
- Near-term targets (LBL, MTLp2)
- Later-term targets (MTLp13-MTLp16)
- Best individual performance: LightGBM on LBL (R² = 0.8522)

#### Moderate Predictions (0.70 < R² < 0.80)
- Mid-term targets (MTLp3, MTLp4)
- MTLp9-MTLp12
- Consistent across all models

#### Challenging Predictions (R² < 0.70)
- MTLp5-MTLp8
- Particularly MTLp8 showing lowest performance
- Area for potential improvement

## Features

- Multi-output regression modeling
- Support for multiple algorithms
- Automated model comparison
- Model export functionality
- Comprehensive evaluation metrics (RMSE, R²)
- Standardized data preprocessing

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- joblib
- liac-arff
- streamlit
- fastapi
- uvicorn
- pydantic

## Project Structure

```
├── data/
│   ├── Supply Chain Management_train.arff
│   └── Supply Chain Management_test.arff
├── exported_models/
│   ├── model_metadata.json
│   ├── random_forest_model.joblib
│   ├── random_forest_scaler.joblib
│   ├── random_forest_config.json
│   ├── xgboost_model.joblib
│   ├── xgboost_scaler.joblib
│   ├── xgboost_config.json
│   ├── lightgbm_model.joblib
│   ├── lightgbm_scaler.joblib
│   └── lightgbm_config.json
├── notebooks/
│   └── Supply Chain Multi-Output Prediction.ipynb
├── src/
│   └── scm.py
│   └── app.py
│   └── api.py
├── requirements.txt
└── README.md
```

## Usage

### 1. Basic Usage

```python
from supply_chain_model import SupplyChainModel

# Initialize model
scm = SupplyChainModel()

# Load and prepare data
train_df = scm.load_arff_data('data/Supply Chain Management_train.arff')
test_df = scm.load_arff_data('data/Supply Chain Management_test.arff')

# Train and evaluate models
X_train, y_train = scm.prepare_data(train_df)
X_test, y_test = scm.prepare_data(test_df)

# Create and train models
scm.create_models()
results = scm.train_and_evaluate(X_train, y_train, X_test, y_test, target_names)

# Export models
scm.export_models()
```

### 2. Loading Saved Models

```python
import joblib

# Load model and scaler
model = joblib.load('exported_models/random_forest_model.joblib')
scaler = joblib.load('exported_models/random_forest_scaler.joblib')

# Make predictions
predictions = model.predict(scaler.transform(new_data))
```

### 3. Running FastAPI

To run the FastAPI server, execute the following command:

```bash
  python src/api.py
```

This starts the API at `http://0.0.0.0:8000`. The following endpoints are available:
- `/predict`: Make predictions using a selected model.
- `/train`: Train models with uploaded training and testing data.
- `/models`: List available models.
- `/download/{model_name}`: Download trained model files.

### 4. Running Streamlit

To launch the Streamlit dashboard, execute:

```bash
streamlit run src/app.py
```

## Deployment Result
- Frontend: http://20.255.59.118:8501/
- Backend: http://20.255.59.118:8000/

This starts the dashboard interface for user interaction.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

## Future Improvements

1. Hyperparameter tuning for each model
2. Feature importance analysis
3. Specialized models for challenging prediction windows
4. Ensemble methods for improved performance
5. Additional feature engineering

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [Supply Chain Management dataset]
- Built using scikit-learn, XGBoost, LightGBM frameworks, Streamlit, and FastAPI