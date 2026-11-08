import pandas as pd

from rivercast.fetch.http import request_json

READINGS_URL = "https://environment.data.gov.uk/hydrology/id/measures/{}/readings.json"


def readings(measure: str, start: str, end: str) -> pd.Series:
    """Raw readings in [start, end)."""
    data = request_json("GET", READINGS_URL.format(measure),
                        params={"mineq-date": start, "max-date": end, "_limit": 500_000})
    items = [i for i in data["items"] if i.get("value") is not None]
    index = pd.to_datetime([i["dateTime"] for i in items])
    return pd.Series([float(i["value"]) for i in items], index=index).sort_index()


def hourly(measure: str, start: pd.Timestamp, end: pd.Timestamp, how: str) -> pd.Series:
    """Hourly series fetched one year at a time; `how` is 'mean' (flow) or 'sum' (rain)."""
    edges = list(pd.date_range(start, end, freq="YS")) or [start]
    edges = sorted({start, *edges, end})
    chunks = [readings(measure, a.date().isoformat(), b.date().isoformat())
              for a, b in zip(edges[:-1], edges[1:], strict=True)]
    raw = pd.concat(chunks)
    resampled = raw.resample("h")
    series = resampled.mean() if how == "mean" else resampled.sum(min_count=1)
    return series.reindex(pd.date_range(start, end, freq="h", inclusive="left"))


def hourly_frame(measures: dict[str, str], start, end, how: str) -> pd.DataFrame:
    return pd.DataFrame({name: hourly(m, start, end, how) for name, m in measures.items()})
