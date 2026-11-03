"""Retrieve and validate public TLC yellow-taxi trip records."""
from datetime import date
from pathlib import Path
import requests
import pandas as pd

SOURCE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
ZONES = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "trip_distance"]


def latest_available_month(today: date | None = None, months_back: int = 8, session=None) -> tuple[int, int]:
    """Find the newest published TLC parquet file, allowing for publication lag."""
    today = today or date.today()
    client = session or requests
    start = today.year * 12 + today.month - 1
    for offset in range(months_back + 1):
        index = start - offset
        year, month = divmod(index, 12)
        month += 1
        url = SOURCE.format(year=year, month=month)
        try:
            response = client.head(url, allow_redirects=True, timeout=15)
            if response.status_code == 200:
                return year, month
            if response.status_code in (403, 405):
                # Some servers disallow HEAD, so probe with a small range request.
                response = client.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=15)
                try:
                    if response.status_code in (200, 206):
                        return year, month
                finally:
                    response.close()
        except requests.RequestException:
            continue
    raise RuntimeError("Could not find a published TLC yellow-taxi month. Check network access or specify --year and --month.")


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(tmp, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def load_trips(path: Path, max_rows: int = 50000, seed: int = 42) -> pd.DataFrame:
    if max_rows < 100:
        raise ValueError("max_rows must be at least 100")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path, columns=COLUMNS)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=COLUMNS)
    else:
        raise ValueError("Expected a .csv or .parquet file")
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    return df.reset_index(drop=True)
