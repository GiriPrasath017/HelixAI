import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, MinMaxScaler
from category_encoders import BinaryEncoder, TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

## Preprocessing

# Missing values and outliers
def missingvalsoutlayers(df, ths=3):
    nmdf = df.copy()

    for col in nmdf.columns:
        if nmdf[col].dtype == 'object':
            nmdf[col].fillna(nmdf[col].mode()[0], inplace=True)
        elif nmdf[col].dtype in ['int64', 'float64']:
            if nmdf[col].isnull().sum() == 0:
                continue
            elif nmdf[col].isnull().sum() / len(nmdf) <= 0.05:
                if -1 < nmdf[col].skew() < 1:
                    nmdf[col].fillna(nmdf[col].mean(), inplace=True)
                else:
                    nmdf[col].fillna(nmdf[col].median(), inplace=True)
            elif isinstance(nmdf.index, pd.RangeIndex):
                nmdf[col].fillna(method='ffill', inplace=True)
            else:
                if pd.to_numeric(nmdf[col], errors='coerce').notnull().all():
                    nmdf[col].interpolate(method='linear', inplace=True, limit_direction='both')
                else:
                    nmdf[col].fillna(nmdf[col].mode()[0], inplace=True)

    for col in nmdf.select_dtypes(include=['int64', 'float64']).columns:
        z_scores = np.abs((nmdf[col] - nmdf[col].mean()) / nmdf[col].std())
        nmdf.loc[z_scores > ths, col] = np.nan

    rclos = nmdf.select_dtypes(include=['int64', 'float64']).columns[nmdf.select_dtypes(include=['int64', 'float64']).isnull().any()]
    if not rclos.empty:
        knn_imputer = KNNImputer(n_neighbors=5)
        nmdf[rclos] = knn_imputer.fit_transform(nmdf[rclos])

    return nmdf


# Encoding
def encoding(data, target_col=None):

    edcopy = data.copy()

    categorical_cols = [col for col in edcopy.columns if edcopy[col].dtype == 'object']
    
    if len(categorical_cols) == 0:
        return edcopy
    
    be = True
    for col in categorical_cols:
        if len(edcopy[col].unique()) > 2:
            be = False
            break
    
    if be:
        encoder = BinaryEncoder()
        edcopy = encoder.fit_transform(edcopy)
    elif target_col is not None and target_col in data.columns:
        target_encodable = True
        for col in categorical_cols:
            if len(edcopy[col].unique()) <= 1:
                target_encodable = False
                break
        
        if target_encodable:
            encoder = TargetEncoder()
            edcopy = encoder.fit_transform(edcopy, data[target_col])
        else:
            one_hot_encodable = True
            for col in categorical_cols:
                if len(edcopy[col].unique()) > 10: 
                    one_hot_encodable = False
                    break
            
            if one_hot_encodable:
                edcopy = pd.get_dummies(edcopy, columns=categorical_cols)
            else:
                encoder = LabelEncoder()
                for col in categorical_cols:
                    edcopy[col] = encoder.fit_transform(edcopy[col])
    else:
        encoder = LabelEncoder()
        for col in categorical_cols:
            edcopy[col] = encoder.fit_transform(edcopy[col])
    
    return edcopy


# Scaling
def scaling(data, method='auto'):

    sdcopy = data.copy()
    
    if method == 'auto':
        if (sdcopy.select_dtypes(include=[np.number]).apply(lambda col: col.max() > 1e4)).any():
            method = 'standard'  
        elif (sdcopy.select_dtypes(include=[np.number]).apply(lambda col: col.min() >= 0)).all():
            method = 'min-max' 
        else:
            method = 'robust'  
        
    if method == 'standard':
        scaler = StandardScaler()
        sdcopy[sdcopy.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(sdcopy.select_dtypes(include=[np.number]))
    elif method == 'min-max':
        scaler = MinMaxScaler()
        sdcopy[sdcopy.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(sdcopy.select_dtypes(include=[np.number]))
    elif method == 'robust':
        scaler = RobustScaler()
        sdcopy[sdcopy.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(sdcopy.select_dtypes(include=[np.number]))
    else:
        raise ValueError("Invalid scaling method. Supported methods: 'auto', 'standard', 'min-max', 'robust'.")

    return sdcopy

## Feature Extraction

def featureextraction(data,tcol):
    nsamps, nf = data.shape
    
    if nsamps < 50:
        pca = PCA(n_components=min(nsamps, nf))
        ftdf = pca.fit_transform(data)
        return ftdf
    elif nsamps >= 50 and nsamps < 500:
        kpca = KernelPCA(n_components=min(nsamps // 2, nf), kernel='rbf')
        ftdf = kpca.fit_transform(data)
        return ftdf
    else:
        lda = LinearDiscriminantAnalysis(n_components=min(nsamps // 2, nf))
        ftdf = lda.fit_transform(data, tcol)
        return ftdf
    
## Train Test split

def split_data(data, tcol, test_size=0.3, val_size=0.5, random_state=42):
    X = data.drop(tcol, axis=1)  
    y = data[tcol]  

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size / (1 - test_size),
                                                    random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test

## Train Model

def algoselection(X_train, X_test, y_train, y_test, ptype):

    if ptype == 'classification':
        if len(X_train) < 100:
            model = LogisticRegression()
        elif len(X_train.columns) > 10:
            model = RandomForestClassifier()
        else:
            model = DecisionTreeClassifier()
    elif ptype == 'regression':
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

def evaluate_model(model, X_test, y_test, report_name='model_report.txt', model_filename='model.pkl'):

    y_pred = model.predict(X_test)

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(model, model_filename)

    with open(report_name, 'w') as report_file:
        report_file.write(f'Accuracy: {accuracy:.2f}\n\n')
        report_file.write(f'Classification Report:\n{report_df}\n\n')
        report_file.write(f'Confusion Matrix:\n{conf_matrix}\n')

    report = pd.DataFrame({
        'Accuracy': [accuracy],
        'Classification Report': [report_df],
        'Confusion Matrix': [conf_matrix]
    })

    return report


def process(df, tcol, ptype, mname, rname):  
    mrdf = missingvalsoutlayers(df, ths=3)
    edf = encoding(mrdf, target_col=tcol)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(edf, tcol, test_size=0.3, val_size=0.5, random_state=42)
    model = algoselection(X_train, X_test, y_train, y_test, ptype)
    report = evaluate_model(model, X_test, y_test, report_name=rname, model_filename=mname)  # Corrected argument names here
    print(report)

dpath = input("Enter the dataset file path: ")
tcol = input("Enter the Target column: ")
ptype = input("Enter the problem type (classification or regression): ")
mname = input("Enter the model name: ")
rname = f"{mname}_report.txt"
mfname = f"{mname}.pkl"

data = pd.read_csv(dpath)
process(data, tcol, ptype, mfname, rname)



