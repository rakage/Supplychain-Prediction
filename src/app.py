import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import io

API_URL = "http://localhost:8000"

def main():
    st.title("Supply Chain Prediction System")
    # Made by Raka Luthfi
    st.subheader("Made by Raka Luthfi")

    
    page = st.sidebar.selectbox("Choose a page", ["Predict", "Train Models", "Download Models"])
    
    if page == "Predict":
        show_prediction_page()
    elif page == "Train Models":
        show_training_page()
    else:
        show_download_page()

def show_prediction_page():
    st.header("Make Predictions")
    
    response = requests.get(f"{API_URL}/models")
    available_models = response.json()["models"]
    
    all_predictions = []
    # Example of data using download button from a file
    file_folder = '../data/test.csv'
    file_example = pd.read_csv(file_folder)
    st.download_button(
        label="Download Example",
        data=file_example.to_csv(index=False),
        file_name="example_data.csv",
        mime="text/csv"
    )

    with st.form("prediction_form"):

        uploaded_file = st.file_uploader("Upload CSV with features", type=['csv'])
        selected_model = st.selectbox("Select Model", available_models)
        submitted = st.form_submit_button("Make Prediction")
        
        if submitted and uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            target_cols = ['LBL'] + [f'MTLp{i}' for i in range(2, 17)]
            feature_cols = [col for col in df.columns if col not in target_cols]
            features = df[feature_cols].values.tolist()
            
            for feature in features:
                data = {
                    'features': feature,
                    'model_name': selected_model
                }
                
                response = requests.post(f"{API_URL}/predict", json=data)
                
                if response.status_code == 200:
                    predictions = response.json()["predictions"]
                    all_predictions.append(predictions)
                else:
                    st.error(f"Error making prediction: {response.json()['detail']}")
                    break
            
    if all_predictions:
        results_df = pd.DataFrame(
            all_predictions,
            columns=target_cols
        )
        
        st.subheader("Predictions")
        st.dataframe(results_df)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

def show_model_config():
    st.subheader("Model Configuration")
    
    with st.expander("Random Forest Configuration"):
        rf_config = {
            'rf_n_estimators': st.number_input("Number of estimators", 10, 1000, 100, key='rf_1'),
            'rf_max_features': st.selectbox("Max features", ['sqrt', 'log2', 'auto'], key='rf_2'),
            'rf_min_samples_split': st.number_input("Min samples split", 2, 20, 2, key='rf_3'),
            'rf_min_samples_leaf': st.number_input("Min samples leaf", 1, 10, 1, key='rf_4'),
            'rf_max_depth': st.number_input("Max depth (0 for None)", 0, 100, 0, key='rf_5')
        }
        # Convert max_depth 0 to None
        rf_config['rf_max_depth'] = None if rf_config['rf_max_depth'] == 0 else rf_config['rf_max_depth']
    
    with st.expander("XGBoost Configuration"):
        xgb_config = {
            'xgb_n_estimators': st.number_input("Number of estimators", 10, 1000, 100, key='xgb_1'),
            'xgb_max_depth': st.number_input("Max depth", 1, 20, 6, key='xgb_2'),
            'xgb_learning_rate': st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01, key='xgb_3'),
            'xgb_subsample': st.number_input("Subsample", 0.1, 1.0, 0.8, 0.1, key='xgb_4'),
            'xgb_colsample_bytree': st.number_input("Colsample bytree", 0.1, 1.0, 0.8, 0.1, key='xgb_5')
        }
    
    with st.expander("LightGBM Configuration"):
        lgb_config = {
            'lgb_n_estimators': st.number_input("Number of estimators", 10, 1000, 100, key='lgb_1'),
            'lgb_max_depth': st.number_input("Max depth", 1, 20, 6, key='lgb_2'),
            'lgb_learning_rate': st.number_input("Learning rate", 0.01, 1.0, 0.1, 0.01, key='lgb_3'),
            'lgb_subsample': st.number_input("Subsample", 0.1, 1.0, 0.8, 0.1, key='lgb_4'),
            'lgb_colsample_bytree': st.number_input("Colsample bytree", 0.1, 1.0, 0.8, 0.1, key='lgb_5')
        }
    
    return {**rf_config, **xgb_config, **lgb_config}

def show_training_page():
    st.header("Train New Models")
    
    config = show_model_config()
    
    with st.form("training_form"):
        train_file = st.file_uploader("Upload Training Data (ARFF)", type=['arff'])
        test_file = st.file_uploader("Upload Test Data (ARFF)", type=['arff'])
        submitted = st.form_submit_button("Train Models")
        
        if submitted and train_file is not None and test_file is not None:
            files = {
                'train_file': ('train.arff', train_file, 'application/octet-stream'),
                'test_file': ('test.arff', test_file, 'application/octet-stream')
            }
            
            with st.spinner('Training models... This may take a while.'):
                response = requests.post(
                    f"{API_URL}/train",
                    files=files,
                    json=config
                )
                
                if response.status_code == 200:
                    results = response.json()
                    st.success(f"Training completed!\nBest model: {results['best_model']}")
                    
                    st.subheader("Model Performance")
                    for model_name, metrics in results['model_performance'].items():
                        if model_name != 'overall_r2':
                            st.write(f"\n{model_name} Performance:")
                            metrics_df = pd.DataFrame(metrics).round(4)
                            st.dataframe(metrics_df)
                else:
                    st.error(f"Error during training: {response.json()['detail']}")

def show_download_page():
    st.header("Download Trained Models")
    
    response = requests.get(f"{API_URL}/models")
    available_models = response.json()["models"]
    
    if not available_models:
        st.warning("No trained models available. Please train models first.")
        return
    
    selected_model = st.selectbox("Select Model to Download", available_models)
    
    if st.button("Download Model"):
        with st.spinner("Preparing download..."):
            response = requests.get(f"{API_URL}/download/{selected_model}", stream=True)
            
            if response.status_code == 200:
                st.download_button(
                    label="Click to Download",
                    data=response.content,
                    file_name=f"{selected_model}_package.zip",
                    mime="application/zip"
                )
            else:
                st.error("Error downloading model")

if __name__ == "__main__":
    main()