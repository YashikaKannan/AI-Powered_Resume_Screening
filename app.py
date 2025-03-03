import streamlit as st
import spacy
import os
import subprocess
import PyPDF2
import pandas as pd
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# Ensure spaCy and model are installed
try:
    import spacy
except ImportError:
    subprocess.run(["pip", "install", "spacy"], check=True)

subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

# Load the model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to check grammar and suggest improvements
def check_grammar(text):
    blob = TextBlob(text)
    return blob.correct()

# Function to check eligibility
def check_eligibility(resume_text, criteria):
    doc = nlp(resume_text.lower())
    found_skills = {token.text for token in doc if token.text in criteria["Skills"]}
    experience = any(exp in resume_text.lower() for exp in criteria["Experience"])
    degree = any(deg in resume_text.lower() for deg in criteria["Degree"])
    return found_skills, experience, degree

# Function to screen a resume against a job description
def screen_resume(resume_text, job_description):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return similarity_score

# Function to screen multiple resumes
def screen_multiple_resumes(resumes, job_description):
    results = []
    for resume_text, file_name in resumes:
        score = screen_resume(resume_text, job_description)
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
        st.text(resume_text[:500])  # Show first 500 characters
        
        # Resume Optimization
        st.subheader("Resume Optimization (Grammar Check):")
        optimized_text = check_grammar(resume_text)
        st.text(optimized_text[:500])
        
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
