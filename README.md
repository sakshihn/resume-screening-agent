# Resume Screening Agent

An AI agent that ranks candidate resumes against a job description, combining
math-based similarity scoring with LLM-generated reasoning — built for the
Rooman AI Challenge (24-Hour AI Agent Challenge).

## What it does

Takes a job description and a folder of resumes, and produces a ranked
shortlist. Each candidate gets:
- A **similarity score** (0–100%), computed mathematically via TF-IDF + cosine similarity
- **Matched skills** and **missing skills**, identified by an LLM
- A short **reasoning summary** explaining the fit

The score and the reasoning come from two separate sources on purpose — see
"Design Approach" below.

## Setup

### 1. Install dependencies
```bash
pip install groq python-dotenv pdfplumber python-docx scikit-learn
```

### 2. Get a free Groq API key
- Sign up at [console.groq.com](https://console.groq.com)
- Go to **API Keys** → **Create API Key**
- Copy the key

### 3. Configure your API key
Create a file named `.env` in the project root:
```
GROQ_API_KEY=your_key_here
```

### 4. Add your data
- Place your job description in `jd.txt`
- Place resumes (`.txt`, `.pdf`, or `.docx`) inside the `resumes/` folder

## Running it

```bash
python agent.py
```

This will:
1. Read the job description and every resume in `resumes/`
2. Score each resume against the JD
3. Ask the LLM to identify matched/missing skills and explain the fit
4. Print a ranked shortlist to the terminal
5. Save full results to `ranked_output.json`

## Design Approach

**Why two separate scoring mechanisms?**

The numeric score comes from **TF-IDF + cosine similarity** — a deterministic,
math-based method that converts the JD and resume text into weighted word
vectors and measures their overlap. This score is reproducible and auditable;
running it twice on the same input gives the same number.

The **reasoning, matched skills, and missing skills** come from the LLM
(Llama 3.3 70B via Groq), which is well-suited to reading unstructured text
and explaining *why* something matches — but is not used to generate the
score itself, to avoid the model inventing or exaggerating a number.

This split is the project's main anti-hallucination safeguard: the ranking
you see is never just "the AI's opinion" — it's backed by a number you can
independently verify.

## Tradeoffs & Limitations

- **TF-IDF is keyword-based**, not semantic. It won't recognize that "ML" and
  "machine learning" mean the same thing, or that "led a team" implies
  leadership skills. A production version would likely use sentence
  embeddings (e.g., `sentence-transformers`) for better semantic matching,
  at the cost of more compute and latency.
- **No PDF layout parsing** beyond plain text extraction — resumes with
  complex multi-column layouts or heavy graphics may extract text out of
  order.
- **No de-duplication or resume-quality checks** (e.g., detecting
  incomplete or corrupted files) — a production version would validate
  extracted text isn't empty before scoring.
- **Single JD at a time** — the agent doesn't yet support screening the same
  resume pool against multiple job openings in one run.
- **With more time**, I'd add: semantic embedding-based scoring as a second
  signal alongside TF-IDF, a simple web UI for non-technical recruiters, and
  batch caching so re-running on the same resumes doesn't re-call the LLM.

## Sample Output

See `ranked_output.json` for the full structured output across all sample
resumes, and the terminal output below for a quick view:

```
1. resume1.txt — 43.32%
   Matched: ['Python', 'pandas', 'numpy', 'scikit-learn', 'SQL', ...]
   Missing: ['deploying models', 'cloud platforms', ...]
   Why: Jane Doe's resume matches most of the required skills...

2. resume3.txt — 23.19%
   ...
```

## Project Files

| File | Purpose |
|---|---|
| `agent.py` | Main script — extraction, scoring, LLM reasoning, ranking |
| `jd.txt` | Sample job description |
| `resumes/` | Sample resumes (10 candidates, varied fit levels) |
| `ranked_output.json` | Output from the last run |
| `.env` | Your Groq API key (not committed to Git) |