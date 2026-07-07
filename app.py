from flask import Flask, request, jsonify, render_template
import os, json, tempfile
from groq import Groq
from dotenv import load_dotenv
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
app = Flask(__name__)

SYSTEM_PROMPT = """You are a resume screening assistant. Given a job description and one
candidate resume, evaluate the fit.

Rules:
- Base your judgment ONLY on what's in the resume text and JD text.
- Do not invent skills or experience not mentioned.
- Output valid JSON only, no markdown, no explanation outside the JSON.
- Format:
{
  "matched_skills": [...],
  "missing_skills": [...],
  "reasoning": "2-3 sentence explanation"
}
"""

def extract_text_from_file(file_storage):
    suffix = file_storage.filename.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        file_storage.save(tmp.name)
        tmp_path = tmp.name

    if suffix == "pdf":
        with pdfplumber.open(tmp_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif suffix == "docx":
        doc = Document(tmp_path)
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()

    os.remove(tmp_path)
    return text

def score_resume(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    return round(float(cosine_similarity(vectors[0], vectors[1])[0][0]) * 100, 2)

def get_reasoning(jd_text, resume_text):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"JOB DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}"}
        ]
    )
    return json.loads(response.choices[0].message.content)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/screen", methods=["POST"])
def screen():
    jd_text = request.form.get("jd_text", "")
    files = request.files.getlist("resumes")

    if not jd_text.strip():
        return jsonify({"error": "Job description is required"}), 400
    if not files or len(files) == 0:
        return jsonify({"error": "At least one resume is required"}), 400
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 resumes allowed"}), 400

    results = []
    for file in files:
        resume_text = extract_text_from_file(file)
        score = score_resume(jd_text, resume_text)
        reasoning = get_reasoning(jd_text, resume_text)
        results.append({"candidate": file.filename, "score": score, **reasoning})

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)