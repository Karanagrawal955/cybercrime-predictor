"""Integration tests for the full API workflow."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db import SessionLocal, Base, engine, User, UserRole
from app.synth import generate_all_synthetic_data


@pytest.fixture(scope="module")
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(scope="module")
def db():
    """Create a test database."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    generate_all_synthetic_data()
    yield db
    db.close()


@pytest.fixture
def admin_token(client: TestClient):
    """Get admin JWT token."""
    response = client.post("/auth/login", data={
        "username": "i4c_admin",
        "password": "admin123",
    })
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def state_token(client: TestClient):
    """Get state officer JWT token."""
    response = client.post("/auth/login", data={
        "username": "state_mh",
        "password": "state123",
    })
    assert response.status_code == 200
    return response.json()["access_token"]


def auth_header(token: str):
    return {"Authorization": f"Bearer {token}"}


def test_health_check(client: TestClient):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_login_admin(client: TestClient):
    """Test admin login."""
    response = client.post("/auth/login", data={
        "username": "i4c_admin",
        "password": "admin123",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["role"] == "i4c_admin"


def test_login_invalid_credentials(client: TestClient):
    """Test login with invalid credentials."""
    response = client.post("/auth/login", data={
        "username": "i4c_admin",
        "password": "wrong",
    })
    assert response.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])