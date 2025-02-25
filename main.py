import streamlit as st
import sys
import PyPDF2
import spacy
import pandas as pd
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

# Display Python version for debugging
st.write(f"Python Version: {sys.version}")

# Ensure `en_core_web_sm` is loaded without subprocess
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import en_core_web_sm
    nlp = en_core_web_sm.load()

# Load sentence transformer model
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text if text else "Error: Unable to extract text from this PDF."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to check grammar and suggest improvements
def check_grammar(text):
    blob = TextBlob(text)
    return blob.correct()

# Function to check eligibility based on predefined criteria
def check_eligibility(resume_text, criteria):
    doc = nlp(resume_text.lower())
    found_skills = {token.text for token in doc if token.text in criteria["Skills"]}
    experience = any(exp in resume_text.lower() for exp in criteria["Experience"])
    degree = any(deg in resume_text.lower() for deg in criteria["Degree"])
    return found_skills, experience, degree

# Function to screen a resume against a job description
def screen_resume(resume_text, job_description):
    try:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embedding = model.encode(job_description, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
        return similarity_score
    except Exception as e:
        return f"Error in screening: {str(e)}"

# Function to screen multiple resumes
def screen_multiple_resumes(resumes, job_description):
    results = []
    for resume_text, file_name in resumes:
        score = screen_resume(resume_text, job_description)
        if isinstance(score, float):  # Ensure valid score
            results.append((file_name, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

# Streamlit UI
st.title("AI-Powered Resume Screening System")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
job_description = st.text_area("Enter Job Description")

if uploaded_files:
    resumes = []

    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_pdf(uploaded_file)
        resumes.append((resume_text, uploaded_file.name))
        
        st.subheader(f"Resume: {uploaded_file.name}")
        st.text(resume_text[:500])  # Display first 500 characters

        # Resume Optimization
        st.subheader("Resume Optimization (Grammar Check):")
        optimized_text = check_grammar(resume_text)
        st.text(optimized_text[:500])  # Limit output for UI stability

        # Eligibility Checking
        criteria = {
            "Skills": ["python", "machine learning", "data analysis", "java"],
            "Experience": ["2+ years", "3+ years", "5+ years"],
            "Degree": ["bachelor", "master", "phd"]
        }
        skills_found, experience_match, degree_match = check_eligibility(resume_text, criteria)

        st.subheader("Eligibility Check:")
        st.write(f"Skills Matched: {', '.join(skills_found) if skills_found else 'None'}")
        st.write(f"Experience Requirement Met: {'Yes' if experience_match else 'No'}")
        st.write(f"Degree Requirement Met: {'Yes' if degree_match else 'No'}")

    # Resume Screening for multiple resumes
    if job_description:
        ranked_resumes = screen_multiple_resumes(resumes, job_description)
        st.subheader("Ranked Resume Matches:")
        df = pd.DataFrame(ranked_resumes, columns=["Resume Name", "Similarity Score"])
        st.dataframe(df)
