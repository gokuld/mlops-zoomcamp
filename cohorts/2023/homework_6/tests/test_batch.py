import pandas as pd
from datetime import datetime

from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2022, 1, 1, hour, minute, second)


def test_prepare_data():
    categorical = ["PULocationID", "DOLocationID"]

    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2), dt(1, 10)),
        (1, 2, dt(2, 2), dt(2, 3)),
        (None, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (2, 3, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]

    df = pd.DataFrame(data, columns=columns)

    expected_prepared_data = [
        (-1, -1, dt(1, 2), dt(1, 10), 8.0),
        (1, -1, dt(1, 2), dt(1, 10), 8.0),
        (1, 2, dt(2, 2), dt(2, 3), 1.0),
    ]

    expected_prepared_data_columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "duration",
    ]

    expected_prepared_df = pd.DataFrame(
        expected_prepared_data, columns=expected_prepared_data_columns
    )
    expected_prepared_df[categorical] = (
        expected_prepared_df[categorical].astype("int").astype("str")
    )

    assert prepare_data(df, categorical).equals(expected_prepared_df)
