import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path,index_col=0)
    return df

def handle_missing_values(df):
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns

    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    return df

def encode_categoricals(df):
    categorical_cols = df.select_dtypes(include='object').columns
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore',sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df, encoder

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def preprocess_data(df):
    df = handle_missing_values(df)

    # Check and map 'Risk' column
    if 'Risk' not in df.columns:
        raise KeyError("Column 'Risk' not found in dataset. Available columns: " + str(df.columns.tolist()))
    
    df['Risk'] = df['Risk'].apply(lambda x: 1 if x == 'good' else 0)

    df, encoder = encode_categoricals(df)

    X = df.drop(['Risk'], axis=1,)
    y = df['Risk']

    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler, encoder
