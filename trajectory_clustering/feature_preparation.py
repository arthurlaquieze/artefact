import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from traffic.core import Traffic


def fit_scaler(data, feature_range: tuple[float, float] = (-1, 1)) -> MinMaxScaler:
    """
    Fits a MinMaxScaler to the input data.

    Args:
        data (np.ndarray): The input data to fit the scaler to. Should be of shape (n_samples, n_features).
        feature_range (Tuple[float, float]): The range of the transformed data. Defaults to (-1, 1).

    Returns:
        MinMaxScaler: The fitted MinMaxScaler object.
    """
    scaler = MinMaxScaler(feature_range=feature_range)  # type: ignore
    scaler.fit(data)
    return scaler


def prepare_features(
    traffic: Traffic,
    list_features: list[str],
    scaler: MinMaxScaler,
    flight_ids: list[str] | None = None,
    points_per_flight: int | None = None,
) -> tuple[npt.NDArray[np.float32], list[str]]:
    if flight_ids is not None:
        traffic = traffic[flight_ids]  # type: ignore

    if points_per_flight is None:
        points_per_flight = int(traffic.data.shape[0] / len(traffic))

    X = np.empty((len(traffic), points_per_flight * len(list_features)), dtype=np.float32)
    order_of_flights = []
    for i, flight in enumerate(traffic):
        _X_unscaled = flight.data[list_features].to_numpy()
        _X = scaler.transform(_X_unscaled)
        X[i] = _X.reshape(-1)

        order_of_flights.append(flight.flight_id)

    return X, order_of_flights


def traffic_from_features(
    X: npt.NDArray[np.float32],
    original_traffic: Traffic,
    order_of_flights: list[str],
    list_features: list[str],
    scaler: MinMaxScaler,
    points_per_flight: int | None = None,
) -> Traffic:
    """
    Reconstructs a Traffic object from the features.
    """
    if points_per_flight is None:
        points_per_flight = int(X.shape[1] / len(list_features))

    X = X.copy()
    X = X.reshape((X.shape[0], points_per_flight, len(list_features)))
    for i in range(X.shape[0]):
        X[i] = scaler.inverse_transform(X[i])

    flights_data = []
    for i, (features, flight_id) in enumerate(zip(X, order_of_flights)):
        flight_data = original_traffic[flight_id].data.copy()  # type: ignore
        flight_data[list_features] = features
        flights_data.append(flight_data)

    return Traffic(pd.concat(flights_data))
