import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import arff
import warnings
import os
import json

warnings.filterwarnings('ignore')

class SupplyChainModel:
    def __init__(self):
        self.models = {
            'random_forest': None,
            'xgboost': None,
            'lightgbm': None
        }
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_features': 'sqrt',
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_depth': None,
                'n_jobs': -1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_jobs': -1,
                'random_state': 42
            }
        }

    def load_arff_data(self, file_path):
        """Load and prepare ARFF file data."""
        with open(file_path, 'r') as f:
            data = arff.load(f)
        df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
        return df

    def prepare_data(self, df):
        """Prepare features and targets."""
        target_cols = ['LBL'] + [f'MTLp{i}' for i in range(2, 17)]
        feature_cols = [col for col in df.columns if col not in target_cols]
        
        X = df[feature_cols]
        y = df[target_cols]
        
        return X, y

    def create_models(self):
        """Create different multi-output regression models with configurable parameters."""
        rf_config = self.model_configs['random_forest']
        xgb_config = self.model_configs['xgboost']
        lgb_config = self.model_configs['lightgbm']
        
        rf_base = RandomForestRegressor(
            n_estimators=rf_config['n_estimators'],
            max_features=rf_config['max_features'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            max_depth=rf_config['max_depth'],
            n_jobs=rf_config['n_jobs'],
            random_state=rf_config['random_state']
        )
        
        xgb_base = xgb.XGBRegressor(
            n_estimators=xgb_config['n_estimators'],
            max_depth=xgb_config['max_depth'],
            learning_rate=xgb_config['learning_rate'],
            subsample=xgb_config['subsample'],
            colsample_bytree=xgb_config['colsample_bytree'],
            n_jobs=xgb_config['n_jobs'],
            random_state=xgb_config['random_state']
        )
        
        lgb_base = lgb.LGBMRegressor(
            n_estimators=lgb_config['n_estimators'],
            max_depth=lgb_config['max_depth'],
            learning_rate=lgb_config['learning_rate'],
            subsample=lgb_config['subsample'],
            colsample_bytree=lgb_config['colsample_bytree'],
            n_jobs=lgb_config['n_jobs'],
            random_state=lgb_config['random_state']
        )
        
        self.models['random_forest'] = MultiOutputRegressor(rf_base)
        self.models['xgboost'] = MultiOutputRegressor(xgb_base)
        self.models['lightgbm'] = MultiOutputRegressor(lgb_base)

    def evaluate_model(self, y_true, y_pred, target_names):
        """Evaluate model performance for each target."""
        results = {}
        overall_r2 = 0
        
        for i, target in enumerate(target_names):
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            
            results[target] = {
                'RMSE': rmse,
                'R2': r2
            }
            overall_r2 += r2
        
        results['overall_r2'] = overall_r2 / len(target_names)
        return results

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, target_names):
        """Train and evaluate all models."""
        best_overall_r2 = -np.inf
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_name] = scaler
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Evaluate
            results = self.evaluate_model(y_test.values, y_pred, target_names)
            all_results[model_name] = results
            
            # Track best model
            if results['overall_r2'] > best_overall_r2:
                best_overall_r2 = results['overall_r2']
                self.best_model = model
                self.best_model_name = model_name
        
        return all_results

    def export_models(self, output_dir='exported_models'):
        """Export all trained models, scalers, and configurations."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save metadata about the best model
        metadata = {
            'best_model': self.best_model_name,
            'configurations': self.model_configs
        }
        
        metadata_path = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        for model_name, model in self.models.items():
            # Export model
            model_path = os.path.join(output_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            
            # Export scaler
            scaler_path = os.path.join(output_dir, f'{model_name}_scaler.joblib')
            joblib.dump(self.scalers[model_name], scaler_path)
            
            # Export configuration
            config_path = os.path.join(output_dir, f'{model_name}_config.json')
            with open(config_path, 'w') as f:
                json.dump(self.model_configs[model_name], f)
        
        print(f"\nBest performing model: {self.best_model_name}")
        print(f"All models, scalers, and configurations exported to {output_dir}/")