import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import glob
import os
import pyarrow.dataset as ds

def compute_events(df):
    df = df.copy()
    
    # Compute total available and unavailable bikes
    df["stock"] = df["vm_disponibles"] + df["vae_disponibles"]
    df["unavailable"] = df["vm_indisponibles"] + df["vae_indisponibles"]

    df["delta_total"] = (df["stock"] + df["unavailable"]).diff()
    df["delta_available"] = df["stock"].diff()

    # Identify arrivals and departures
    df["arrivals"] = df["delta_total"].clip(lower=0)
    df["departures"] = (-df["delta_total"]).clip(lower=0)

    # Apply regulation and maintenance constraints
    not_regulated = 1 - df["reg_flag"]
    not_under_maintenance = (df["unavailable"].diff() + 0.1 >= 0).astype(float)

    df["true_departures"] = df["departures"] * not_under_maintenance * not_regulated
    df["true_arrivals"] = df["arrivals"] * not_regulated

    df["available"] = df["delta_available"].clip(lower=0)
    df["unavailable_flag"] = (-df["delta_available"]).clip(lower=0)

    # Generate dock and bike signals
    df["dock_signal"] = None
    df.loc[df["departures"] > 0, "dock_signal"] = 1
    df.loc[df["arrivals"] > 0, "dock_signal"] = -1

    df["bike_signal"] = None
    df.loc[df["available"] > 0, "bike_signal"] = 1
    df.loc[df["unavailable_flag"] > 0, "bike_signal"] = -1

    # Track recent activity for censoring logic
    recent_dock = df["dock_signal"].astype(float).ffill().shift(1).fillna(0) > 0
    recent_bike = df["bike_signal"].astype(float).ffill().shift(1).fillna(0) > 0
    df["fresh_dock"] = recent_dock | (df["arrivals"] > 0)
    df["fresh_bike"] = recent_bike | (df["true_departures"] > 0)

    return df


def apply_censoring(df, min_stock=7, min_ranges=5):
    df = df.copy()

    # Censor observations when station is too empty or too full
    df["censor_empty"] = (df["stock"] < min_stock) & (~df["fresh_bike"])
    df["censor_full"] = (df["diapason_disponibles"] < min_ranges) & (~df["fresh_dock"])

    df["obs_departure"] = df["true_departures"]
    df.loc[df["censor_empty"], "obs_departure"] = np.nan

    df["obs_arrival"] = df["true_arrivals"]
    df.loc[df["censor_full"], "obs_arrival"] = np.nan

    return df


def resample_to(df, freq="1min"):
    # Resample time series to uniform intervals
    df.index = pd.to_datetime(df.index)
    df = df.groupby(df.index).first()
    return df.resample(freq).ffill()


def gaussian_filter_nan(x, sigma=5):
    # Smooth data with Gaussian filter while preserving NaNs
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    x_filled = np.nan_to_num(x, nan=0.0)
    num = gaussian_filter1d(x_filled, sigma=sigma, mode="nearest")
    den = gaussian_filter1d(mask.astype(float), sigma=sigma, mode="nearest")
    out = num / (den + 1e-8)
    out[den <= 0.1] = 0
    out[mask] = x[mask]
    out = np.nan_to_num(out)
    return out

def read_metadata():
    data_path = "instant_updates/"  # folder containing all station subfolders

    # Step 1: List all stations by scanning folder names
    station_dirs = [f.split("/")[-1] for f in glob.glob(os.path.join(data_path, "*")) if os.path.isdir(f)]
    nstations = len(station_dirs)
    print(f"Found {nstations} stations")

    # Step 2: Use first station to get feature columns
    first_station_file = os.path.join(data_path, station_dirs[0], "time_serie.parquet")
    if os.path.exists(first_station_file):
        df_first = ds.dataset(first_station_file).to_table().to_pandas()
        feature_cols = df_first.columns.tolist()
        print(f"Found {len(feature_cols)} columns")
    else:
        raise FileNotFoundError(f"No parquet file found for first station: {station_dirs[0]}")
    
    begin = pd.Timestamp("2023-01-01")
    end = (pd.Timestamp("2025-01-01") - pd.Timedelta("1h"))
    nhours = int((end - begin) / pd.Timedelta("1h")) + 1

    default_row = {feat: pd.NA for feat in feature_cols}
    begin_row = pd.DataFrame([default_row], index=[begin])
    end_row = pd.DataFrame([default_row], index=[end])

    # Step 3: Build meta dictionary
    meta = {
        "stations": station_dirs,
        "features": feature_cols,
        "begin": begin_row,
        "end" : end_row,
        "nstations": nstations,
        "nhours": nhours,
    }
    return(meta)

def timeserie(st, meta):
    parquet_file = f"instant_updates/{st}/time_serie.parquet"
    df = ds.dataset(parquet_file).to_table().to_pandas()

    df.index = pd.to_datetime(df.index)

    df = pd.concat([meta["begin"], df, meta["end"]])

    return df