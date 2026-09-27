"""
app.py
------
Simple Streamlit UI for the Resume Analyzer & Job Matcher.

Run with:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os

from resume_parser import parse_resume, parse_txt
from matcher import match_resume_to_job

st.set_page_config(page_title="Resume Analyzer & Job Matcher", page_icon="📄", layout="wide")

st.title("📄 Resume Analyzer & Job Matcher")
st.caption("Pure NLP pipeline: tokenization → stopword removal → lemmatization → TF-IDF → cosine similarity. No LLMs involved.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], key="resume")
    resume_text_input = st.text_area("...or paste resume text", height=200, key="resume_text")

with col2:
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload job description", type=["pdf", "docx", "txt"], key="jd")
    jd_text_input = st.text_area("...or paste job description text", height=200, key="jd_text")


def get_text(uploaded_file, pasted_text):
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            if suffix.lower() == ".txt":
                return parse_txt(tmp_path)
            return parse_resume(tmp_path)
        finally:
            os.unlink(tmp_path)
    return pasted_text or ""


if st.button("Analyze Match", type="primary"):
    resume_text = get_text(resume_file, resume_text_input)
    jd_text = get_text(jd_file, jd_text_input)

    if not resume_text.strip() or not jd_text.strip():
        st.error("Please provide both a resume and a job description (upload or paste).")
    else:
        with st.spinner("Running NLP pipeline..."):
            result = match_resume_to_job(resume_text, jd_text)

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Match Score", f"{result['overall_match_score']}%")
        m2.metric("TF-IDF Text Similarity", f"{result['tfidf_similarity']}%")
        m3.metric("Skill Coverage", f"{result['skill_match_percent']}%")

        st.divider()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### ✅ Matched Skills")
            st.write(", ".join(result["matched_skills"]) or "None found")
        with c2:
            st.markdown("### ❌ Missing Skills")
            st.write(", ".join(result["missing_skills"]) or "None — great coverage!")
        with c3:
            st.markdown("### ➕ Extra Skills (not in JD)")
            st.write(", ".join(result["extra_skills_not_in_jd"]) or "None")

        if result["experience_note"]:
            st.divider()
            st.markdown("### 🕒 Experience Check")
            st.info(result["experience_note"])

        st.divider()
        k1, k2 = st.columns(2)
        with k1:
            st.markdown("### 🔑 Top Resume Keywords (TF-IDF)")
            st.write(", ".join(result["top_resume_keywords"]))
        with k2:
            st.markdown("### 🔑 Top JD Keywords (TF-IDF)")
            st.write(", ".join(result["top_jd_keywords"]))

        with st.expander("Raw JSON output"):
            st.json(result)
