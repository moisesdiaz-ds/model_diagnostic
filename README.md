# model_diagnostic
Get a full diagnostic of a trained model: 
- Feature importance
- Perfomance metrics
- Feature distributions
- Model performance by features values
- False negatives profiling
- Global Shap values
- Shap partial depence plots
- Local shap values plots

# How it works
The framework uses the tracking_models module (Already on the repo), so its necesary to have your model saved by the tracking_models.save_model function

In order to get the diagnostic you only need to specify the following parameters on the model_diagnostic function:

- X_train: Data the model used for the training phase

- y_train: Target data the model used for the training phase

- X_test: Data the model used for the test phase

- y_test: Target data the model used for the test phase

- dir_results_files: Directory from the tracking models package where the metrics, stats and other files are stored.

- dir_models: Directory from the tracking models package where the models are stored.

- model_id: Model id you want to diagnose

- model_id2: Model id you want to compare

- umbral_elegido: Chosen threshold for making the predictions

- limit_imp: Qty of features youn want to see on the feature importance section 

- model_name: Name how you identify the model (Can be anything) 

- model_name2: Name how you identify the 2nd model (Can be anything) 

- data_for_comparison: Data you want to use for comparing with the training data distribution 

- return_model_metrics: If you want to show the model metrics

- return_feature_importance_gini: If you want to show the  feature importance using gini method

- return_compare_data_dist: If you want to show features distribution compared with other data

- return_performance_by_segment: If you want to show the model performance by segment

- return_profiling_false_negatives: If you want to show the model metrics false negatives profiling

- return_shap: If you want to show the model shap analysis

- return_feature_importance_permutation: If you want to show the  model feature importance by permutation


# Example


