from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import joblib
import os
from scm import SupplyChainModel
import arff
import io
from fastapi.responses import FileResponse
import zipfile
import tempfile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
# CORS
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelConfig(BaseModel):
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_features: str = 'sqrt'
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_depth: Optional[int] = None
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # LightGBM parameters
    lgb_n_estimators: int = 100
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8

class PredictionInput(BaseModel):
    features: List[float]
    model_name: str 

class TrainingResponse(BaseModel):
    model_performance: Dict
    best_model: str

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_used: str

class FeatureMetadata(BaseModel):
    feature_names: List[str]
    feature_descriptions: Dict[str, str]

def load_model(model_name):
    models_dir = "../exported_models"
    
    # Determine which model to load
    if model_name == "lightgbm":
        model_name = "lightgbm"
    elif model_name == "xgboost":
        model_name = "xgboost"
    elif model_name == "random_forest":
        model_name = "random_forest"
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler, model_name

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput):
    try:
        model, scaler, model_name = load_model(input_data.model_name)
        features = np.array(input_data.features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        
        return PredictionResponse(
            predictions=predictions[0].tolist(),
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    config: ModelConfig = ModelConfig()
):
    try:
        scm = SupplyChainModel()
        
        # Configure models with user parameters
        scm.model_configs = {
            'random_forest': {
                'n_estimators': config.rf_n_estimators,
                'max_features': config.rf_max_features,
                'min_samples_split': config.rf_min_samples_split,
                'min_samples_leaf': config.rf_min_samples_leaf,
                'max_depth': config.rf_max_depth,
                'n_jobs': -1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': config.xgb_n_estimators,
                'max_depth': config.xgb_max_depth,
                'learning_rate': config.xgb_learning_rate,
                'subsample': config.xgb_subsample,
                'colsample_bytree': config.xgb_colsample_bytree,
                'n_jobs': -1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': config.lgb_n_estimators,
                'max_depth': config.lgb_max_depth,
                'learning_rate': config.lgb_learning_rate,
                'subsample': config.lgb_subsample,
                'colsample_bytree': config.lgb_colsample_bytree,
                'n_jobs': -1,
                'random_state': 42
            }
        }
        
        # Read ARFF files
        train_content = await train_file.read()
        test_content = await test_file.read()
        
        train_data = arff.loads(train_content.decode('utf-8'))
        test_data = arff.loads(test_content.decode('utf-8'))
        
        train_df = pd.DataFrame(train_data['data'], 
                              columns=[attr[0] for attr in train_data['attributes']])
        test_df = pd.DataFrame(test_data['data'], 
                             columns=[attr[0] for attr in test_data['attributes']])
        
        X_train, y_train = scm.prepare_data(train_df)
        X_test, y_test = scm.prepare_data(test_df)
        
        scm.create_models()
        results = scm.train_and_evaluate(X_train, y_train, X_test, y_test, y_train.columns)
        scm.export_models(output_dir="../exported_models")
        
        return TrainingResponse(
            model_performance=results,
            best_model=scm.best_model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_available_models():
    models_dir = "../exported_models"
    if not os.path.exists(models_dir):
        return {"models": []}
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith("_model.joblib"):
            models.append(file.replace("_model.joblib", ""))
    
    return {"models": models}

@app.get("/download/{model_name}")
async def download_model(model_name: str):
    models_dir = "../exported_models"
    model_path = os.path.join(models_dir, f"{model_name}_model.joblib")
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.joblib")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
        with zipfile.ZipFile(tmp.name, 'w') as archive:
            archive.write(model_path, f"{model_name}_model.joblib")
            archive.write(scaler_path, f"{model_name}_scaler.joblib")
    
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename=f"{model_name}_package.zip"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)