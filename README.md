# HerHealthGPT — FemSympQA dataset pipeline

Multilingual (EN/FR/AR) symptom→condition/risk/action dataset built from NHS/CDC/NIH
sources. Spec: `docs/superpowers/specs/2026-07-05-femsympqa-dataset-pipeline-design.md`.

## Setup
1. Python 3.12, then: `python -m venv .venv`
2. `.\.venv\Scripts\python -m pip install -e .[dev]`
3. Copy `.env.example` to `.env` and add your Anthropic API key.

## Pipeline run order
(filled in as stages land — see scripts/)
