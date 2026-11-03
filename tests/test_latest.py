from datetime import date
from routerisk.data import latest_available_month


class Reply:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeSession:
    def __init__(self, published):
        self.published = published
        self.checked = []

    def head(self, url, **kwargs):
        self.checked.append(url)
        return Reply(200 if url in self.published else 404)


def test_latest_month_with_publication_delay():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-08.parquet"
    client = FakeSession({url})
    assert latest_available_month(date(2026, 11, 2), session=client) == (2026, 8)
    assert len(client.checked) == 4


def test_latest_month_crosses_year_boundary():
    url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-11.parquet"
    client = FakeSession({url})
    assert latest_available_month(date(2026, 1, 7), session=client) == (2025, 11)
