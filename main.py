"""
main.py
-------
Command-line entry point.

Usage:
    python main.py --resume resume.pdf --jd job_description.txt
    python main.py --resume resume.docx --jd jd.txt --json
"""

import argparse
import json
import os

from resume_parser import parse_resume, parse_txt
from matcher import match_resume_to_job, format_report


def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return parse_txt(path)
    return parse_resume(path)


def main():
    parser = argparse.ArgumentParser(description="Resume Analyzer & Job Matcher (pure NLP)")
    parser.add_argument("--resume", required=True, help="Path to resume file (.pdf/.docx/.txt)")
    parser.add_argument("--jd", required=True, help="Path to job description file (.pdf/.docx/.txt)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of a formatted report")
    args = parser.parse_args()

    resume_text = load_text(args.resume)
    jd_text = load_text(args.jd)

    result = match_resume_to_job(resume_text, jd_text)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
