import pandas as pd

from sklearn.feature_extraction import DictVectorizer

from prefect import task, flow


@task
def load_data(filename, t_min=1, t_max=60):
    df = pd.read_parquet(filename)

    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])

    df['duration'] = df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

    df = df[(df['duration'] >= t_min) & (df['duration'] <= t_max)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df

@task
def add_features(data_train, data_val):
    data_train['PU_DO'] = data_train['PULocationID'] + '_' + \
                          data_train['DOLocationID']
    data_val['PU_DO'] = data_val['PULocationID'] + '_' + \
                        data_val['DOLocationID']

    return data_train, data_val

@task
def encode_data(data_train, data_val):
    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = data_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = data_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    y_train = data_train['duration'].values
    y_val = data_val['duration'].values

    return X_train, X_val, y_train, y_val, dv

@flow
def preprocess_data(train_path, val_path):
    """Combines all the functions above in a single step."""

    # Read and preprocess data
    train_data = load_data(train_path)
    val_data = load_data(val_path)

    # Add features
    X_train, X_val = add_features(train_data, val_data)

    # Encoding data
    X_train, X_val, y_train, y_val, dv = encode_data(X_train, X_val)

    return X_train, X_val, y_train, y_val, dv


if __name__ == "__main__":
    preprocess_data()
