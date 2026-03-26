import argparse
from datetime import date

from loguru import logger
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


def read_dataframe(filename: str) -> pd.DataFrame:
    """
    Reads a parquet file into a pandas DataFrame, processes the data, and returns a cleaned DataFrame.

    Returns a DataFrame with the following transformations applied:
        - A 'duration' column representing the ride duration in minutes (calculated as the difference
          between 'lpep_dropoff_datetime' and 'lpep_pickup_datetime').
        - The DataFrame is filtered to only include rows where 'duration' is between 1 and 60 minutes.
        - The columns 'PULocationID' and 'DOLocationID' are converted to string data type.

    Parameters:
    -----------
    filename : str
        The path to the parquet file to be read.

    Returns:
    --------
    pd.DataFrame
        The transformed df
        

    Notes:
    ------
    - The function assumes that the parquet file contains 'lpep_pickup_datetime' and 'lpep_dropoff_datetime' columns,
      which are used to compute the 'duration'.
    - The output DataFrame will not contain rows where the calculated 'duration' is outside the range [1, 60] minutes.
    """
    try:
        df = pd.read_parquet(filename)

        df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df.duration = df.duration.dt.total_seconds() / 60

        df = df[(df.duration >= 1) & (df.duration <= 60)]

        categorical = ['PULocationID', 'DOLocationID']
        df[categorical] = df[categorical].astype(str)
        
        return df
    except Exception as e:
        logger.error(f"ERROR: reading {filename} failed")
        logger.error(e)
        raise


def train(train_date: date, val_date: date, out_path: str) -> None:
    """
    Trains a linear regression model on taxi trip data and saves the trained pipeline.

    Parameters:
    -----------
    train_date : date
        The date for the training data (used to construct the file URL).
    val_date : date
        The date for the validation data (used to construct the file URL).
    out_path : str
        The path where the trained model pipeline will be saved.

    Returns:
    --------
    None
        The function trains the model and saves the pipeline to the specified output path.
    """
    base_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    train_url = base_url.format(year=train_date.year, month=train_date.month)
    val_url = base_url.format(year=val_date.year, month=val_date.month)

    df_train = read_dataframe(train_url)
    df_val = read_dataframe(val_url)

    logger.info(f"df_train length is: {len(df_train)}")
    logger.info(f"df_val length is: {len(df_val)}")

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']


    target = 'duration'
    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    y_train = df_train[target].values
    y_val = df_val[target].values

    dv = DictVectorizer()
    lr = LinearRegression()
    pipeline = make_pipeline(dv, lr)

    pipeline.fit(train_dicts, y_train)
    y_pred = pipeline.predict(val_dicts)

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"MSE: {mse}")


    with open(out_path, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model based on specified dates and save it to a given path")
    parser.add_argument("--train-date", required=True, help="train month in the YYYY-MM format")
    parser.add_argument("--val-date", required=True, help="val month in the YYYY-MM format")
    parser.add_argument("--model-save-path", required=True, help="Path where the trained model should be saved.")
    
    args = parser.parse_args()
    train_year, train_month = args.train_date.split("-")
    val_year, val_month = args.val_date.split("-")

    train_date = date(int(train_year), int(train_month), 1)
    val_date = date(int(val_year), int(val_month), 1)
    out_path = args.model_save_path
    train(train_date=train_date, val_date=val_date, out_path=out_path)
