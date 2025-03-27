import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import joblib
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import mlflow
import mlflow.sklearn

# Function to load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess the data
def preprocess_data(df):
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Drop Date and Location columns
    df.drop(["Date", "Location"], axis="columns", inplace=True)

    # One-Hot Encoding for categorical variables
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    ohe_cols = enc.fit_transform(df[["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]])

    # Concatenate encoded features and drop original categorical columns
    df = pd.concat([df, ohe_cols], axis=1)
    df.drop(["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"], axis=1, inplace=True)

    # Convert target variable to binary
    df['RainTomorrow'] = df['RainTomorrow'].replace({'No': 0, 'Yes': 1})

    return df

# Function to split data into train and test sets
def split_data(df):
    y = df['RainTomorrow']
    X = df.drop(['RainTomorrow'], axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Function to perform PCA
def perform_pca(X_train_scaled, X_test_scaled):
    pca = PCA()
    pca.fit(X_train_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.90) + 1
    print(f"Number of components to keep: {n_components}")

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca, n_components

# Function to train the XGBoost model
def train_xgboost(X_train_pca, y_train, X_test_pca, y_test):
    n_cpus = multiprocessing.cpu_count()
    parameters = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 300, 500, 1000], 'max_depth': [3, 5, 7]}
    xgboost = xgb.XGBClassifier(objective='binary:logistic', random_state=0)

    clf = GridSearchCV(xgboost, parameters, n_jobs=n_cpus)
    clf.fit(X_train_pca, y_train)

    xgboost = xgb.XGBClassifier(learning_rate=clf.best_params_['learning_rate'], 
                                max_depth=clf.best_params_['max_depth'], 
                                n_estimators=clf.best_params_['n_estimators'], 
                                subsample=0.8, colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=0.1, 
                                objective='binary:logistic', random_state=0)
    xgboost.fit(X_train_pca, y_train)

    xgboost_y_pred_prob = xgboost.predict_proba(X_test_pca)[:,1]
    xgboost_y_pred = xgboost.predict(X_test_pca)

    print('XGBoost Accuracy:', accuracy_score(y_test, xgboost_y_pred))
    print('XGBoost Classification Report:\n', classification_report(y_test, xgboost_y_pred))

    # Confusion Matrix
    ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, xgboost_y_pred), display_labels=xgboost.classes_).plot()
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, xgboost_y_pred_prob)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1], [0,1], 'k--')
    plt.title('XGBoost ROC Curve')
    plt.show()
    print('XGBoost ROC AUC:', roc_auc_score(y_test, xgboost_y_pred_prob))

    return xgboost

# Function to save the trained model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved successfully as {filename}")

def main():
    # Set up MLflow experiment
    mlflow.set_experiment("WeatherPrediction_XGBoost")
    
    with mlflow.start_run():
        # Load and preprocess the data
        df = load_data("weatherAUS.csv")
        df = preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = split_data(df)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Perform PCA
        X_train_pca, X_test_pca, n_components = perform_pca(X_train_scaled, X_test_scaled)

        # Train the XGBoost model
        model = train_xgboost(X_train_pca, y_train, X_test_pca, y_test)

        # Log model and metrics with MLflow
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test_pca)))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, model.predict_proba(X_test_pca)[:, 1]))

        # Save the model locally
        save_model(model, "xgboost_model.pkl")

if __name__ == "__main__":
    main()

