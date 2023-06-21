#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import pandas as pd
import click


# Parameters
output_directory = "./output/"

# Load model, data and make predictions

with open("model.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

categorical = ["PULocationID", "DOLocationID"]


def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")

    return df


def predict_durations(year, month):
    df = read_data(
        f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    )

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    df_result = df.loc[:, ["ride_id"]].copy()
    df_result["duration_prediction"] = y_pred
    return df_result


@click.command()
@click.option("--year", "-y", required=True, type=click.IntRange(min=0000, max=9999))
@click.option("--month", "-m", required=True, type=click.IntRange(min=1, max=12))
def predict_durations_and_save_output(year, month):
    df_result = predict_durations(year, month)

    # print the standard deviation of predictions
    print(
        f"Standard deviation of predictions: {df_result['duration_prediction'].std()}"
    )

    # Save predictions to file
    output_uri = os.path.join(
        output_directory,
        f"duration_predictions_yellow_tripdata_{year:04d}_{month:02d}.parquet",
    )
    print(f"Saving predictions to {output_uri}")
    df_result.to_parquet(output_uri, engine="pyarrow", compression=None, index=False)


if __name__ == "__main__":
    predict_durations_and_save_output()
