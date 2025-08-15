#!/usr/bin/env python3
"""
Auto Email Summarizer & Categorizer (Ollama + LLaMA 3.2 4B)
Usage:
  python main.py --file email.txt
  python main.py --input "Email body..."
"""
import argparse, requests, json, sys, os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "llama3.2:4b"
TIMEOUT = 300

def run_llama(prompt):
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()

def build_prompt(email_text):
    return (
        "You are an assistant that summarizes and tags emails.\n"
        "Task:\n"
        "1) Provide an EXACT 2-sentence summary of the email.\n"
        "2) Provide one-line next step.\n"
        "3) Output a tag on its own line: Tag: Urgent | Follow-Up | Informational\n\n"
        f"Email:\n{email_text}\n\n"
        "Respond in this exact format:\nSummary: <two sentences>\nNext step: <one sentence>\nTag: <one of Urgent|Follow-Up|Informational>"
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", help="Inline email text")
    p.add_argument("--file", "-f", help="Path to file with email text")
    args = p.parse_args()
    content = args.input or ""
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as fh:
                content = (content + "\n" if content else "") + fh.read()
        except Exception as e:
            print("Error reading file:", e, file=sys.stderr); sys.exit(1)
    if not content.strip():
        print("Provide --input or --file", file=sys.stderr); sys.exit(1)
    prompt = build_prompt(content)
    out = run_llama(prompt)
    print(out)

if __name__ == "__main__":
    main()
