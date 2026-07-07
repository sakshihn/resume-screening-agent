import sys
import os
import io
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app import app, score_resume


# --- Tests for score_resume() ---
# This is the deterministic TF-IDF + cosine similarity function, so it's
# fully testable without hitting the Groq API.

def test_identical_text_scores_high():
    jd = "Python developer with experience in machine learning and data analysis"
    resume = "Python developer with experience in machine learning and data analysis"
    score = score_resume(jd, resume)
    assert score > 90

def test_unrelated_text_scores_low():
    jd = "Python developer with machine learning experience"
    resume = "Professional chef specializing in Italian cuisine and pastry"
    score = score_resume(jd, resume)
    assert score < 20

def test_partial_match_scores_moderate():
    jd = "Looking for a Python developer with SQL and cloud experience"
    resume = "Experienced Python developer, strong SQL skills, no cloud background"
    score = score_resume(jd, resume)
    assert 20 < score < 90

def test_score_is_between_0_and_100():
    jd = "Data scientist role requiring Python, SQL, and machine learning"
    resume = "Marketing manager with 5 years of social media experience"
    score = score_resume(jd, resume)
    assert 0 <= score <= 100

def test_empty_resume_scores_zero():
    jd = "Python developer with machine learning experience"
    resume = ""
    score = score_resume(jd, resume)
    assert score == 0.0


# --- Tests for the /screen route's input validation ---
# These don't reach the Groq API call since they fail validation first.

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

def test_screen_missing_jd_returns_400(client):
    data = {"resumes": (io.BytesIO(b"dummy resume text"), "resume.txt")}
    response = client.post("/screen", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_screen_missing_resumes_returns_400(client):
    data = {"jd_text": "Python developer needed"}
    response = client.post("/screen", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()

def test_screen_too_many_resumes_returns_400(client):
    data = {
        "jd_text": "Python developer needed",
    }
    data["resumes"] = [
        (io.BytesIO(b"resume text"), f"resume{i}.txt") for i in range(11)
    ]
    response = client.post("/screen", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "error" in response.get_json()