import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
import multiprocessing

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

    return xgboost

# Function to save the trained model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved successfully as {filename}")

