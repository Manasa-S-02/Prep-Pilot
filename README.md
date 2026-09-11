# 🚀 PrepPilot — AI Mock Interview Coach

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manasa-S-02/Prep-Pilot/blob/main/Prep_Pilot.ipynb)

An AI interviewer that asks role-specific questions and **grades your answers with an NLI cross-encoder** — not by just asking a language model to guess a number.

![demo](demo.gif)

## What it does

Pick a role, and PrepPilot runs a 5-question mock interview mixing technical and behavioral questions. After each answer you get instant, explainable feedback showing which key points you covered and missed; at the end you get a full report with an overall score, strengths, and what to work on.

## How it works

1. **Question generation** — an LLM produces role-specific, non-repeating questions.
2. **Multi-turn interview** — a stateful 5-question flow with running feedback and a final report.
3. **Answer scoring (the core)** — the ideal answer is decomposed into atomic *key points*, and an **NLI cross-encoder** (`cross-encoder/nli-deberta-v3-base`) checks whether the candidate's answer *entails* each one. The score is the mean entailment, and the UI shows exactly which points were covered vs missed. This entailment-based, rubric-style evaluation is the answer-scoring method I built during my internship.

## Tech

Python · Gradio · Groq (LLM) · Sentence-Transformers cross-encoder · NumPy

## Run it

**Easiest:** click the *Open in Colab* badge above and run the cells (add your own free Groq key).

**Locally:**
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here     # free key: https://console.groq.com/keys
python app.py
```

## Roadmap

- Multilingual interviews (English / Tamil / Hindi)
- Job-description-tailored questions (RAG)
- Voice answers (speech-to-text)

## Author

**Manasa S** — [GitHub](https://github.com/Manasa-S-02)
First-author, ACL LT-EDI 2026 (multilingual span detection & counter-narrative generation).
