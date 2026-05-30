"""Performance + concurrency checks against the targets (<200 ms, ~10 concurrent)."""
import concurrent.futures as cf
import time

from tests.conftest import CANON


def test_warm_latency_under_200ms(client):
    client.post("/predict", json=CANON)  # warm the model + caches
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        assert client.post("/predict", json=CANON).status_code == 200
    avg_ms = (time.perf_counter() - t0) / n * 1000
    assert avg_ms < 200, f"average latency {avg_ms:.1f} ms exceeds 200 ms"


def test_handles_concurrent_requests(client):
    def one(_):
        return client.post("/predict", json=CANON).status_code

    with cf.ThreadPoolExecutor(max_workers=10) as ex:
        codes = list(ex.map(one, range(30)))
    assert all(c == 200 for c in codes), f"non-200 responses: {[c for c in codes if c != 200]}"
