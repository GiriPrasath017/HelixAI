import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

## Data Visualization

def visualize_data(df, x_col, y_col):
    if df[x_col].dtype == 'object' and df[y_col].dtype == 'object':
        sns.countplot(x=x_col, hue=y_col, data=df)
        plt.title(f'Count of {y_col} by {x_col}')
        plt.xlabel(x_col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title=y_col)
        plt.show()

    elif df[x_col].dtype == 'object' and df[y_col].dtype in ['int64', 'float64']:
        sns.boxplot(x=x_col, y=y_col, data=df)
        plt.title(f'{y_col} by {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xticks(rotation=45)
        plt.show()

    elif df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
        plt.scatter(df[x_col], df[y_col])
        plt.title(f'{y_col} vs. {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    else:
        print("Data types not supported for visualization.")


## Preprocessing

# Missing values and outliers
def handle_missing_outliers(df, threshold=3):
    df_copy = df.copy()

    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
        elif df_copy[col].dtype in ['int64', 'float64']:
            if df_copy[col].isnull().sum() == 0:
                continue
            elif df_copy[col].isnull().sum() / len(df_copy) <= 0.05:
                if -1 < df_copy[col].skew() < 1:
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                else:
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)
            elif isinstance(df_copy.index, pd.RangeIndex):
                df_copy[col].fillna(method='ffill', inplace=True)
            else:
                if pd.to_numeric(df_copy[col], errors='coerce').notnull().all():
                    df_copy[col].interpolate(method='linear', inplace=True, limit_direction='both')
                else:
                    df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)

    for col in df_copy.select_dtypes(include=['int64', 'float64']).columns:
        z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
        df_copy.loc[z_scores > threshold, col] = np.nan

    rclos = df_copy.select_dtypes(include=['int64', 'float64']).columns[df_copy.select_dtypes(include=['int64', 'float64']).isnull().any()]
    if not rclos.empty:
        knn_imputer = KNNImputer(n_neighbors=5)
        df_copy[rclos] = knn_imputer.fit_transform(df_copy[rclos])

    return df_copy

# Encoding
def encode_data(data, target_col=None):

    data_copy = data.copy()
    categorical_cols = [col for col in data_copy.columns if data_copy[col].dtype == 'object']
    
    if len(categorical_cols) == 0:
        return data_copy
    
    binary_encodable = all(len(data_copy[col].unique()) <= 2 for col in categorical_cols)
    if binary_encodable:
        encoder = BinaryEncoder()
        data_copy = encoder.fit_transform(data_copy)
    elif target_col is not None and target_col in data.columns:
        target_encodable = all(len(data_copy[col].unique()) > 1 for col in categorical_cols)
        if target_encodable:
            encoder = TargetEncoder()
            data_copy = encoder.fit_transform(data_copy, data[target_col])
        else:
            one_hot_encodable = all(len(data_copy[col].unique()) <= 10 for col in categorical_cols)
            if one_hot_encodable:
                data_copy = pd.get_dummies(data_copy, columns=categorical_cols)
            else:
                encoder = LabelEncoder()
                for col in categorical_cols:
                    data_copy[col] = encoder.fit_transform(data_copy[col])
    else:
        encoder = LabelEncoder()
        for col in categorical_cols:
            data_copy[col] = encoder.fit_transform(data_copy[col])
    
    return data_copy

# Scaling
def scale_data(data, method='auto'):

    data_copy = data.copy()
    
    if method == 'auto':
        if (data_copy.select_dtypes(include=[np.number]).apply(lambda col: col.max() > 1e4)).any():
            method = 'standard'  
        elif (data_copy.select_dtypes(include=[np.number]).apply(lambda col: col.min() >= 0)).all():
            method = 'min-max' 
        else:
            method = 'robust'  
        
    if method == 'standard':
        scaler = StandardScaler()
        data_copy[data_copy.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(data_copy.select_dtypes(include=[np.number]))
    elif method == 'min-max':
        scaler = MinMaxScaler()
        data_copy[data_copy.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(data_copy.select_dtypes(include=[np.number]))
    elif method == 'robust':
        scaler = RobustScaler()
        data_copy[data_copy.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(data_copy.select_dtypes(include=[np.number]))
    else:
        raise ValueError("Invalid scaling method. Supported methods: 'auto', 'standard', 'min-max', 'robust'.")

    return data_copy

## Feature Extraction

def extract_features(data, target_col):
    n_samples, n_features = data.shape
    
    if n_samples < 50:
        pca = PCA(n_components=min(n_samples, n_features))
        feature_data = pca.fit_transform(data)
    elif 50 <= n_samples < 500:
        kpca = KernelPCA(n_components=min(n_samples // 2, n_features), kernel='rbf')
        feature_data = kpca.fit_transform(data)
    else:
        lda = LinearDiscriminantAnalysis(n_components=min(n_samples // 2, n_features))
        feature_data = lda.fit_transform(data, target_col)
    
    return feature_data
    
## Train Test split

    def split_data(data, target_col, test_size=0.3, val_size=0.5, random_state=42):
        X = data.drop(target_col, axis=1)  
        y = data[target_col]  

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size / (1 - test_size),
                                                        random_state=random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    ## Train Model

    def select_model(X_train, X_test, y_train, y_test, problem_type):

        if problem_type == 'classification':
            if len(X_train) < 100:
                model = LogisticRegression()
            elif len(X_train.columns) > 10:
                model = RandomForestClassifier()
            else:
                model = DecisionTreeClassifier()
        elif problem_type == 'regression':
            if len(X_train) < 100:
                model = LinearRegression()
            elif len(X_train.columns) > 10:
                model = RandomForestRegressor()
            else:
                model = DecisionTreeRegressor()
        else:
            raise ValueError("Invalid problem type. Please specify 'classification' or 'regression'.")

        scaler = StandardScaler()
        numerical_columns = X_train.select_dtypes(include=['number']).columns
        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        model.fit(X_train, y_train)

        return model

    ## Model Prediction and Evaluation

    def evaluate_model(model, X_test, y_test, encoding_method, scaling_method, algorithm, report_name='model_report.txt', model_filename='model.pkl'):

        y_pred = model.predict(X_test)

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model, model_filename)

        with open(report_name, 'w') as report_file:
            report_file.write(f'Encoding Method: {encoding_method}\n')
            report_file.write(f'Scaling Method: {scaling_method}\n')
            report_file.write(f'Algorithm Used: {algorithm}\n\n')
            report_file.write(f'Accuracy: {accuracy:.2f}\n\n')
            report_file.write(f'Classification Report:\n{report_df}\n\n')
            report_file.write(f'Confusion Matrix:\n{conf_matrix}\n')

        report = pd.DataFrame({
            'Encoding Method': [encoding_method],
            'Scaling Method': [scaling_method],
            'Algorithm Used': [algorithm],
            'Accuracy': [accuracy],
            'Classification Report': [report_df],
            'Confusion Matrix': [conf_matrix]
        })

        return report

    def process(df, tcol, ptype, mname, rname):  
        mrdf = handle_missing_outliers(df, threshold=3)
        edf = encode_data(mrdf, target_col=tcol)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(edf, tcol, test_size=0.3, val_size=0.5, random_state=42)
        model = select_model(X_train, X_test, y_train, y_test, ptype)
        report = evaluate_model(model, X_test, y_test, encoding_method='auto', scaling_method='standard', algorithm=ptype, report_name=rname, model_filename=mname)
        print(report)

    dpath = input("Enter the dataset file path: ")
    tcol = input("Enter the Target column: ")
    ptype = input("Enter the problem type (classification or regression): ")
    mname = input("Enter the model name: ")
    rname = f"{mname}_report.txt"
    mfname = f"{mname}.pkl"

    data = pd.read_csv(dpath)
    process(data, tcol, ptype, mfname, rname)

