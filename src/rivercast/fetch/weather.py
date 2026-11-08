import pandas as pd

from rivercast.fetch.http import request_json

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def hourly_weather(lon: float, lat: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """ERA5-based hourly air temperature (C) and shortwave radiation (W/m2), UTC."""
    data = request_json("GET", ARCHIVE_URL, params={
        "latitude": lat, "longitude": lon, "timezone": "UTC",
        "start_date": start.date().isoformat(),
        "end_date": (end - pd.Timedelta(hours=1)).date().isoformat(),
        "hourly": "temperature_2m,shortwave_radiation",
    })
    h = data["hourly"]
    df = pd.DataFrame({"air_temp": h["temperature_2m"], "radiation": h["shortwave_radiation"]},
                      index=pd.to_datetime(h["time"]))
    return df.reindex(pd.date_range(start, end, freq="h", inclusive="left"))
