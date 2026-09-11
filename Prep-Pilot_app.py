"""
PrepPilot — AI Mock Interview Coach
-----------------------------------
An AI interviewer that asks role-specific questions and grades your answers.

How it works (3 layers):
  1. An LLM generates role-specific interview questions (technical + behavioral).
  2. A multi-turn interview flow (5 questions) with running feedback and a final report.
  3. Answer scoring via an NLI cross-encoder: the ideal answer is decomposed into
     atomic "key points", and each answer is checked for whether it *entails* them.
     (This is the answer-evaluation method I built during my internship.)

Setup:
  - Set your GROQ_API_KEY as an environment variable (locally) or a Space secret (on Hugging Face).
  - Get a free key at https://console.groq.com/keys  (no credit card).

Run locally:
  pip install -r requirements.txt
  export GROQ_API_KEY=your_key_here
  python app.py
"""

import os
import re
import json
import numpy as np
import gradio as gr
from openai import OpenAI
from sentence_transformers import CrossEncoder

# LLM backend (Groq free tier by default)
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY  = os.environ.get("GROQ_API_KEY", "")
MODEL    = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY or "missing-key")

# NLI cross-encoder — the scoring engine (loads once at startup)
nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")  # labels: [contradiction, entailment, neutral]

TOTAL_QUESTIONS = 5


def llm(system, user):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# Layer 1: question generation
def make_question(role, asked):
    asked_txt = "\n".join(f"- {q}" for q in asked) if asked else "(none yet)"
    system = ("You are a senior interviewer running a mock interview. Ask ONE new interview "
              "question for the role. Vary between technical and behavioral. Do NOT repeat any "
              "already-asked question. Output only the question.")
    return llm(system, f"Role: {role}\nAlready asked:\n{asked_txt}")


# Layer 3: NLI cross-encoder scoring
def generate_key_points(role, question):
    system = ("You are an expert interviewer. List the 3-5 essential points an ideal answer to the "
              "question MUST cover. Return ONLY a JSON array of short strings. No other text.")
    raw = llm(system, f"Role: {role}\nQuestion: {question}")
    txt = raw.strip().strip("`").strip()
    if txt.lower().startswith("json"):
        txt = txt[4:].strip()
    try:
        pts = json.loads(txt)
        if isinstance(pts, list) and pts:
            return [str(p) for p in pts][:5]
    except Exception:
        pass
    lines = [re.sub(r'^[\s\-\*\d\.\)]+', '', l).strip() for l in raw.splitlines() if l.strip()]
    return [l for l in lines if l][:5] or ["A clear, correct, relevant answer"]


def nli_score(answer, key_points):
    logits = np.array(nli.predict([(answer, kp) for kp in key_points]))
    results = [(kp, float(softmax(row)[1])) for kp, row in zip(key_points, logits)]  # [1] = entailment
    score = round(sum(e for _, e in results) / len(results) * 10, 1)
    covered = [f"\u2713 {kp}  ({e:.2f})" for kp, e in results if e >= 0.5]
    missed  = [f"\u2717 {kp}  ({e:.2f})" for kp, e in results if e < 0.5]
    lines = [f"**Score:** {score}/10  _(NLI cross-encoder vs {len(results)} key points)_"]
    if covered:
        lines.append("**Covered:**\n" + "\n".join(covered))
    if missed:
        lines.append("**Missed:**\n" + "\n".join(missed))
    return "\n\n".join(lines)


def score_one(role, question, answer):
    return nli_score(answer, generate_key_points(role, question))


# Layer 2: interview flow
def build_report(session):
    lines = [f"Q{i}: {t['q']}\nAnswer: {t['a']}\nNote: {t['feedback']}"
             for i, t in enumerate(session["transcript"], 1)]
    system = ("You are an interview coach writing a final report. Based on the whole interview, give:\n"
              "**Overall:** X/10\n**Top strengths:** 2-3 bullets\n"
              "**Work on this:** 2-3 bullets\n**One thing to do next:** 1 line")
    return llm(system, f"Role: {session['role']}\n\nInterview:\n" + "\n\n".join(lines))


def q_display(q, idx):
    return f"### Question {idx} of {TOTAL_QUESTIONS}\n{q}"


def start_interview(role):
    role = (role or "Software Engineer").strip()
    try:
        q1 = make_question(role, [])
    except Exception as e:
        return f"\u26a0\ufe0f {e}", "", "", None
    session = {"role": role, "idx": 1, "current_q": q1, "transcript": []}
    return q_display(q1, 1), "", "", session


def submit_answer(answer, session):
    if not session:
        return "Click **Start interview** first.", "", "", session
    if not (answer or "").strip():
        return q_display(session["current_q"], session["idx"]), answer, "", session
    try:
        fb = score_one(session["role"], session["current_q"], answer)
    except Exception as e:
        return q_display(session["current_q"], session["idx"]), answer, f"\u26a0\ufe0f {e}", session
    session["transcript"].append({"q": session["current_q"], "a": answer, "feedback": fb})
    if session["idx"] < TOTAL_QUESTIONS:
        asked = [t["q"] for t in session["transcript"]]
        try:
            nxt = make_question(session["role"], asked)
        except Exception as e:
            return q_display(session["current_q"], session["idx"]), "", f"\u26a0\ufe0f {e}", session
        session["idx"] += 1
        session["current_q"] = nxt
        return q_display(nxt, session["idx"]), "", f"*Last answer* \u2014\n\n{fb}", session
    try:
        report = build_report(session)
    except Exception as e:
        report = f"\u26a0\ufe0f {e}"
    return "### \u2705 Interview complete \u2014 see your report below.", "", report, session


# UI
with gr.Blocks(title="PrepPilot") as demo:
    gr.Markdown("# \U0001f680 PrepPilot \u2014 AI Mock Interview Coach")
    gr.Markdown("A 5-question mock interview. Each answer is graded by an NLI cross-encoder "
                "against the key points an ideal answer should cover.")
    with gr.Row():
        role = gr.Textbox(label="Role", value="Software Engineer", scale=3)
        start_btn = gr.Button("Start interview", variant="primary", scale=1)
    question_box = gr.Markdown("*Click Start interview to begin.*")
    session = gr.State(None)
    answer = gr.Textbox(label="Your answer", lines=6, placeholder="Type your answer, then click Submit...")
    submit_btn = gr.Button("Submit answer", variant="primary")
    report_box = gr.Markdown("")

    start_btn.click(start_interview, inputs=role,
                    outputs=[question_box, answer, report_box, session])
    submit_btn.click(submit_answer, inputs=[answer, session],
                     outputs=[question_box, answer, report_box, session])

if __name__ == "__main__":
    demo.launch()
