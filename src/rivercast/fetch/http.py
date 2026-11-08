import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import requests

USER_AGENT = "RiverCast/0.1 (research prototype)"
RETRY_STATUS = {429, 502, 503, 504}


def request_json(method: str, url: str, retries: int = 5, backoff_s: float = 5.0,
                 timeout_s: float = 180, **kwargs) -> Any:
    for attempt in range(retries):
        try:
            r = requests.request(method, url, headers={"User-Agent": USER_AGENT},
                                 timeout=timeout_s, **kwargs)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in RETRY_STATUS:
                r.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(backoff_s * 2**attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}")


def cached_json(path: Path, fetch: Callable[[], Any]) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    data = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def cached_frame(path: Path, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df
