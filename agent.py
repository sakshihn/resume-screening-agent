from groq import Groq
from dotenv import load_dotenv
import os
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


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


def extract_text(filepath):
    if filepath.endswith(".pdf"):
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif filepath.endswith(".docx"):
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

def score_resume(jd_text, resume_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([jd_text, resume_text])
    return round(float(cosine_similarity(vectors[0], vectors[1])[0][0]) * 100, 2)


import json

def run(jd_path, resumes_folder):
    jd_text = extract_text(jd_path)
    results = []

    for fname in os.listdir(resumes_folder):
        filepath = os.path.join(resumes_folder, fname)
        resume_text = extract_text(filepath)
        score = score_resume(jd_text, resume_text)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"JOB DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}"}
            ]
        )
        reasoning = json.loads(response.choices[0].message.content)

        results.append({
            "candidate": fname,
            "score": score,
            **reasoning
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

if __name__ == "__main__":
    ranked = run("jd.txt", "resumes")

    for i, r in enumerate(ranked, 1):
        print(f"\n{i}. {r['candidate']} — {r['score']}%")
        print(f"   Matched: {r['matched_skills']}")
        print(f"   Missing: {r['missing_skills']}")
        print(f"   Why: {r['reasoning']}")

    with open("ranked_output.json", "w") as f:
        json.dump(ranked, f, indent=2)
    print("\nSaved full results to ranked_output.json")

# jd_text = extract_text("jd.txt")
# resume_text = extract_text("resumes/resume1.txt")

# score = score_resume(jd_text, resume_text)

# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": f"JOB DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}"}
#     ]
# )

# print(f"Similarity Score: {score}%")
# print(response.choices[0].message.content)


# # temporary test - we'll remove this once we add real resumes
# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "JOB DESCRIPTION: Need a Python developer with SQL skills.\n\nRESUME: I know Python, SQL, and have 2 years experience."}
#     ]
# )

# print(response.choices[0].message.content)