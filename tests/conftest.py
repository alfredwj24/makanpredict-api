"""Shared pytest fixtures."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

CANON = {"item": "AYAM BERSIH - STANDARD", "premise_type": "Pasar Basah", "state": "Sabah"}


@pytest.fixture(scope="session")
def client():
    """A TestClient with the app's lifespan run once (model loaded a single time)."""
    with TestClient(app) as c:
        yield c
