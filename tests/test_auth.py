"""Tests for authentication and role-based access control."""

import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException

from app.auth import (
    hash_password, verify_password, create_access_token, decode_token,
    get_current_user, require_roles, filter_by_user_scope
)
from app.db import SessionLocal, Base, engine, User, UserRole, Complaint
from app.synth import generate_all_synthetic_data


@pytest.fixture(scope="module")
def db():
    """Create a test database with synthetic data."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    generate_all_synthetic_data()
    yield db
    db.close()


def test_password_hashing():
    """Test password hashing and verification."""
    password = "test_password_123"
    hashed = hash_password(password)
    
    assert hashed != password
    assert verify_password(password, hashed) == True
    assert verify_password("wrong_password", hashed) == False


def test_jwt_token_creation():
    """Test JWT token creation and decoding."""
    token = create_access_token({"sub": "testuser", "role": "i4c_admin"})
    
    assert isinstance(token, str)
    assert len(token) > 0
    
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"
    assert payload["role"] == "i4c_admin"


def test_invalid_token_decoding():
    """Test that invalid tokens fail to decode."""
    payload = decode_token("invalid.token.here")
    assert payload is None
    
    payload = decode_token("")
    assert payload is None


def test_role_based_dependencies():
    """Test role-based dependency factory."""
    admin_user = User(username="admin", role=UserRole.i4c_admin)
    state_user = User(username="state", role=UserRole.state_officer, state="Maharashtra")
    district_user = User(username="district", role=UserRole.district_officer, state="Delhi", district="Delhi")
    bank_user = User(username="bank", role=UserRole.bank_nodal, bank="HDFC")
    
    require_admin = require_roles(UserRole.i4c_admin)
    require_state = require_roles(UserRole.i4c_admin, UserRole.state_officer)
    require_district = require_roles(UserRole.i4c_admin, UserRole.state_officer, UserRole.district_officer)
    require_bank = require_roles(UserRole.i4c_admin, UserRole.bank_nodal)


def test_filter_by_user_scope_admin(db: Session):
    """Test that admin sees all complaints."""
    admin_user = User(role=UserRole.i4c_admin)
    
    query = db.query(Complaint)
    filtered = filter_by_user_scope(query, Complaint, admin_user)
    
    all_count = query.count()
    filtered_count = filtered.count()
    assert filtered_count == all_count


def test_filter_by_user_scope_state_officer(db: Session):
    """Test that state officer sees only their state."""
    state_user = User(role=UserRole.state_officer, state="Maharashtra")
    
    query = db.query(Complaint)
    filtered = filter_by_user_scope(query, Complaint, state_user)
    
    all_count = query.count()
    filtered_count = filtered.count()
    assert filtered_count <= all_count
    
    results = filtered.all()
    for c in results:
        assert c.victim_state == "Maharashtra"


def test_filter_by_user_scope_district_officer(db: Session):
    """Test that district officer sees only their district."""
    district_user = User(role=UserRole.district_officer, state="Delhi", district="Delhi")
    
    query = db.query(Complaint)
    filtered = filter_by_user_scope(query, Complaint, district_user)
    
    results = filtered.all()
    for c in results:
        assert c.victim_city == "Delhi"


def test_filter_by_user_scope_bank_nodal(db: Session):
    """Test that bank nodal officer sees only their bank."""
    bank_user = User(role=UserRole.bank_nodal, bank="HDFC")
    
    query = db.query(Complaint)
    filtered = filter_by_user_scope(query, Complaint, bank_user)
    
    results = filtered.all()
    for c in results:
        assert c.victim_bank == "HDFC"


def test_demo_users_exist(db: Session):
    """Test that demo users are created."""
    users = db.query(User).all()
    usernames = [u.username for u in users]
    
    expected_users = [
        "i4c_admin", "state_mh", "state_dl", "state_ka",
        "district_delhi_central", "district_mumbai",
        "bank_hdfc", "bank_sbi", "bank_icici"
    ]
    
    for expected in expected_users:
        assert expected in usernames


def test_demo_user_roles(db: Session):
    """Test that demo users have correct roles."""
    admin = db.query(User).filter(User.username == "i4c_admin").first()
    assert admin.role == UserRole.i4c_admin
    
    state = db.query(User).filter(User.username == "state_mh").first()
    assert state.role == UserRole.state_officer
    assert state.state == "Maharashtra"
    
    district = db.query(User).filter(User.username == "district_delhi_central").first()
    assert district.role == UserRole.district_officer
    assert district.state == "Delhi"
    assert district.district == "Central Delhi"
    
    bank = db.query(User).filter(User.username == "bank_hdfc").first()
    assert bank.role == UserRole.bank_nodal
    assert bank.bank == "HDFC"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])