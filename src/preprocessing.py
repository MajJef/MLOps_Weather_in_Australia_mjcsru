import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

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

# Function to standardize the data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

