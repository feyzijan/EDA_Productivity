import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_predict, LeaveOneGroupOut
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd  # Needed if your data inclu

from data_prep_helper import *


features_to_drop = {'mixed_dc_term',
    'mixed_dynamic_range_feat',
    'mixed_first_derivative_std_feat',
    'mixed_mean_feat',
    'mixed_spectral_energy',
    'mixed_sum_of_all_coefficients',
    'phasic_absolute_slope_feat',
    'phasic_dynamic_range_feat',
    'phasic_first_derivative_std_feat',
    'phasic_first_derivetive_mean_feat',
    'phasic_slope_feat',
    'phasic_std_feat',
    'phasic_sum_of_all_coefficients',
    'tonic_absolute_slope_feat',
    'tonic_dc_term',
    'tonic_dynamic_range_feat',
    'tonic_first_derivetive_mean_feat',
    'tonic_max_feat',
    'tonic_mean_feat',
    'tonic_min_feat',
    'tonic_spectral_energy',
    'tonic_sum_of_all_coefficients'}


def load_data(folder):
    X = pd.read_csv(f"{folder}/X.csv", index_col=0).to_numpy()
    X_pca = pd.read_csv(f"{folder}/X_pca.csv", index_col=0)
    X_pruned = pd.read_csv(f"{folder}/X_pruned.csv", index_col=0).to_numpy()
    y = pd.read_csv(f"{folder}/y.csv", index_col=0).to_numpy().flatten()
    return X, X_pca, X_pruned, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def read_combined_dataset():
    X, X_pca, X_pruned, y = load_data("ModelDatasets/a3_a4_combined")
    return split_data(X, y), split_data(X_pca, y)
    # , split_data(X_pruned, y)

def read_a3_dataset():
    X_a3, X_a3_pca, X_a3_pruned, y_a3 = load_data("ModelDatasets/a3")
    return split_data(X_a3, y_a3), split_data(X_a3_pca, y_a3)
# , split_data(X_a3_pruned, y_a3)

def read_a4_dataset():
    X_a4, X_a4_pca, X_a4_pruned, y_a4 = load_data("ModelDatasets/a4")
    return split_data(X_a4, y_a4), split_data(X_a4_pca, y_a4)
    # split_data(X_a4_pruned, y_a4)


def load_each_subject_individually_modelling():
    X_list = []
    y_list = []

    print("now laoding a3 participants")

    p_list_a3_use = [p.split("_")[0] for p in p_list_a3]
    p_list_a3_use = list(set(p_list_a3_use))

    for p in p_list_a3_use:
        folder_path = f"ModelDatasets/{p}/a3"
        X = pd.read_csv(f"{folder_path}/x.csv") 
        y = pd.read_csv(f"{folder_path}/y.csv").to_numpy()
        X_list.append(X)
        y_list.append(y)
        print(f"loaded {p} a3, x shape is {X.shape}, y shape is {y.shape}")

    print("now laoding a4 participants")

    p_list_a4_use = [p.split("_")[0] for p in p_list_a4]
    p_list_a4_use = list(set(p_list_a4_use))
    for p in p_list_a4_use:
        folder_path = f"ModelDatasets/{p}/a4"
        X = pd.read_csv(f"{folder_path}/x.csv") 
        y = pd.read_csv(f"{folder_path}/y.csv").to_numpy()

        X_list.append(X)
        y_list.append(y)
        print(f"loaded {p} a4, x shape is {X.shape}, y shape is {y.shape}")

    pca = joblib.load('pca_a3_a4_model.joblib')

    X_list_pca = [ pca.transform(df) for df in X_list]
    X_list_pruned = [ df.drop(columns=features_to_drop).to_numpy() for df in X_list]
    X_list = [ df.to_numpy() for df in X_list]

    return X_list, X_list_pca, X_list_pruned, y_list


''' Add function to read each subject individually '''
# X_list_train, X_list_test, y_list_train, y_list_test = [], [], [], []
# for X, y in zip(X_list, y_list):
#     X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_list_train.append(X_train_temp)
#     X_list_test.append(X_test_temp)
#     y_list_train.append(y_train_temp.flatten())
#     y_list_test.append(y_test_temp.flatten())

# X_list_pca_train, X_list_pca_test, y_list_pca_train, y_list_pca_test = [], [], [], []
# for X, y in zip(X_list_pca, y_list):
#     X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_list_pca_train.append(X_train_temp)
#     X_list_pca_test.append(X_test_temp)
#     y_list_pca_train.append(y_train_temp.flatten())
#     y_list_pca_test.append(y_test_temp.flatten())

# X_list_pruned_train, X_list_pruned_test, y_list_pruned_train, y_list_pruned_test = [], [], [], []
# for X, y in zip(X_list_pruned, y_list):
#     X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_list_pruned_train.append(X_train_temp)
#     X_list_pruned_test.append(X_test_temp)
#     y_list_pruned_train.append(y_train_temp.flatten())
#     y_list_pruned_test.append(y_test_temp.flatten())


def run_random_forest_with_search(
    X_train, X_test, y_train, y_test, 
    param_grid=None, use_random_search=False, n_iter=50, cv=3, 
    scoring='f1_macro', random_state=42):
    
    if param_grid is None:
        # Define default parameter grid if not
        param_grid = {
            'min_samples_split': np.arange(2, 11, 2),  # Values from 2 to 10, steps of 2
            'min_samples_leaf': np.arange(1, 6),  # Values from 1 to 5
            'max_features': ['sqrt', 'log2']  # Options for max_features
        }
    
    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=random_state)
    
    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=n_iter,  # Number of random samples to draw
        cv=cv,          # Number of cross-validation folds
        n_jobs=-1,      # Use all available cores
        verbose=0,
        random_state=random_state
    )
    
    # Fit RandomizedSearchCV on the training data
    print("Fitting Random Forest with Randomized Search...")
    random_search.fit(X_train, y_train)
    
    # Best hyperparameters and model
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    
    y_pred_test = best_model.predict(X_test)
    
    # Classification report
    clf_report_test = classification_report(y_test, y_pred_test)
    
    # Print results
    print("Best Hyperparameters:", best_params)
    print("Test Set Classification Report:\n", clf_report_test)
    print("Test Set Accuracy Score:", accuracy_score(y_test, y_pred_test))
    print("Test Set F1 Score Weighted:", f1_score(y_test, y_pred_test, average='weighted'))
    print("Test Set F1 Score Macro:", f1_score(y_test, y_pred_test, average='macro'))
    
    return best_model, best_params, clf_report_test


def run_xgboost_with_search(
    X_train, X_test, y_train, y_test, 
    param_grid=None, use_random_search=False, n_iter=50, cv=3, 
    scoring='f1_macro', random_state=42):
    
    if param_grid is None:
        # Define default parameter grid if not provided
        param_grid = { 
            'n_estimators': [50, 200],
            'max_depth': [5, 10],
            'min_child_weight': [3, 4],
            'subsample': [0.75, 0.8]
        }

    # Calculate the scale_pos_weight for imbalanced dataset
    class_count_0 = np.sum(y_train == 0) * 2
    class_count_1 = np.sum(y_train == 1)
    scale_pos_weight = class_count_0 / class_count_1

    # Initialize the classifier
    xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=random_state)

    # Choose the search method
    if use_random_search:
        random_param_grid =  {
            'n_estimators': randint(param_grid['n_estimators'][0], param_grid['n_estimators'][-1]),
            'max_depth': randint(param_grid['max_depth'][0], param_grid['max_depth'][-1]),
            'min_child_weight': randint(param_grid['min_child_weight'][0], param_grid['min_child_weight'][-1]),
            'subsample': (param_grid['subsample'][0], param_grid['subsample'][-1]-param_grid['subsample'][0])  
        }
        search = RandomizedSearchCV(
            estimator=xgb, 
            param_distributions=random_param_grid, 
            n_iter=n_iter, 
            cv=cv, 
            n_jobs=-1, 
            verbose=1, 
            random_state=random_state,
            scoring=scoring
        )
    else:
        search = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            verbose=1, 
            scoring=scoring
        )

    # Fit search on the training data
    print("Fitting XGBoost with Search...")
    search.fit(X_train, y_train)

    # Best hyperparameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_

    # Results
    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    clf_report_train = classification_report(y_train, y_pred_train)
    clf_report_test = classification_report(y_test, y_pred_test)
    print("Best Hyperparameters:", best_params)
    print("Train Set Classification Report:\n", clf_report_train)
    print("Test Set Classification Report:\n", clf_report_test)
    print("Test Set Accuracy Score:", accuracy_score(y_test, y_pred_test))
    print("Test Set F1 Score Macro:", f1_score(y_test, y_pred_test, average='macro'))

    return best_model, best_params, clf_report_test


'''
Given a list of data for each partipant, does CV on a participant leve
'''
def run_xgboost_with_leave_one_subject_out(
    X_list, y_list, 
    param_grid=None, use_random_search=False, n_iter=50, 
    scoring='f1_macro', random_state=42):

    # Concatenate the data
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Create groups array
    groups = []
    for idx, x in enumerate(X_list):
        groups.extend([idx]*len(x))
    groups = np.array(groups)
    
    if param_grid is None:
        # Define default parameter grid if not provided
        param_grid = { 
            'n_estimators': [50, 200],
            'max_depth': [5, 10],
            'min_child_weight': [3, 4],
            'subsample': [0.75, 0.8]
        }
    
    # Calculate the scale_pos_weight for imbalanced dataset
    class_count_0 = np.sum(y == 0) * 2
    class_count_1 = np.sum(y == 1)
    scale_pos_weight = class_count_0 / class_count_1
    
    # Initialize the classifier
    xgb = XGBClassifier(
        eval_metric='logloss', 
        scale_pos_weight=scale_pos_weight, 
        random_state=random_state
    )
    
    # Define the cross-validation strategy
    logo = LeaveOneGroupOut()
    
    # Choose the search method
    if use_random_search:
        # Adjust random_param_grid accordingly
        random_param_grid =  {
            'n_estimators': randint(param_grid['n_estimators'][0], param_grid['n_estimators'][-1]+1),
            'max_depth': randint(param_grid['max_depth'][0], param_grid['max_depth'][-1]+1),
            'min_child_weight': randint(param_grid['min_child_weight'][0], param_grid['min_child_weight'][-1]+1),
            'subsample': uniform(param_grid['subsample'][0], param_grid['subsample'][-1]-param_grid['subsample'][0])  
        }
        search = RandomizedSearchCV(
            estimator=xgb, 
            param_distributions=random_param_grid, 
            n_iter=n_iter, 
            cv=logo.split(X, y, groups=groups), 
            n_jobs=-1, 
            verbose=1, 
            random_state=random_state,
            scoring=scoring
        )
    else:
        search = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid, 
            cv=logo.split(X, y, groups=groups), 
            n_jobs=-1, 
            verbose=1, 
            scoring=scoring
        )
    
    # Fit search on the data
    print("Fitting XGBoost with Leave-One-Subject-Out Cross-Validation...")
    search.fit(X, y, groups=groups)
    
    # Best hyperparameters and model
    best_params = search.best_params_
    best_model = search.best_estimator_
    
    y_pred = cross_val_predict(best_model, X, y, cv=logo.split(X, y, groups=groups), n_jobs=-1)
    
    clf_report = classification_report(y, y_pred)
    print("Best Hyperparameters:", best_params)
    print("Classification Report:\n", clf_report)
    print("Accuracy Score:", accuracy_score(y, y_pred))
    print("F1 Score Macro:", f1_score(y, y_pred, average='macro'))
    
    return best_model, best_params, clf_report



'''
SVC
'''
def run_svc_grid_search(X_train, X_test, y_train, y_test, param_grid_svc=None, cv=2, verbose=2, random_state=42):
    
    # Define default parameter grid if not provided
    if param_grid_svc is None:
        param_grid_svc = {
            "C": [1.0],
            "kernel": ["rbf"],
            "degree": [3],
            "gamma": ["scale"],
            "shrinking": [True]
        }
    
    svc = SVC(random_state=random_state)
    # GridSearchCV
    grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=cv, n_jobs=-1, verbose=verbose)
    print("Fitting SVM with Grid Search...")
    grid_search_svc.fit(X_train, y_train)
    
    # Best hyperparameters and model
    best_params = grid_search_svc.best_params_
    best_model = grid_search_svc.best_estimator_

    # Results
    y_pred_test = best_model.predict(X_test)
    clf_report_test = classification_report(y_test, y_pred_test)
    print("Best Hyperparameters for SVC:", best_params)
    print("Best Estimator for SVC:", best_model)
    print("Test Set Classification Report:\n", clf_report_test)
    
    return best_model, best_params, clf_report_test

# best_model_svc, best_params_svc, clf_report_test_svc = run_svc_grid_search(X_train, X_test, y_train, y_test)


'''
Log reg
'''
def run_logistic_regression_grid_search(X_train, X_test, y_train, y_test, param_grid_lr=None, cv=2, verbose=2, random_state=42):
    
    if param_grid_lr is None:
        param_grid_lr = {
            "penalty": ["l2"],
            "C": [0.001, 0.01],
            "solver": ["newton-cg"],
            "max_iter": [100, 200],
            "multi_class": ["auto"]
        }
    
    logreg = LogisticRegression(random_state=random_state)
    
    # GridSearchCV
    grid_search_lr = GridSearchCV(estimator=logreg, param_grid=param_grid_lr, cv=cv, n_jobs=-1, verbose=verbose)
    print("Fitting Logistic Regression with Grid Search...")
    grid_search_lr.fit(X_train, y_train)
    
    # Best hyperparameters and model
    best_params = grid_search_lr.best_params_
    best_model = grid_search_lr.best_estimator_
    
    # Results
    y_pred_test = best_model.predict(X_test)
    clf_report_test = classification_report(y_test, y_pred_test)
    print("Best Hyperparameters for Logistic Regression:", best_params)
    print("Best Estimator for Logistic Regression:", best_model)
    print("Test Set Classification Report:\n", clf_report_test)
    
    return best_model, best_params, clf_report_test

# best_model_lr, best_params_lr, clf_report_test_lr = run_logistic_regression_grid_search(X_train, X_test, y_train, y_test)


def plot_feature_importances(best_model, col_names):
    feature_importances = best_model.feature_importances_

    # Visualize these in a bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')

    plt.xticks(ticks=range(len(feature_importances)), labels=col_names, rotation=90)
    plt.show()
