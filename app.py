import streamlit as st
import PyPDF2
import pandas as pd
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util

# Load model
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
    resume_text_lower = resume_text.lower()
    found_skills = {skill for skill in criteria["Skills"] if skill.lower() in resume_text_lower}
    experience = any(re.search(rf"\b{exp}\b", resume_text_lower) for exp in criteria["Experience"])
    degree = any(re.search(rf"\b{deg}\b", resume_text_lower) for deg in criteria["Degree"])
    return found_skills, experience, degree

# Function to screen a resume against a job description
def screen_resume(resume_text, job_description):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return similarity_score

# Function to screen multiple resumes
def screen_multiple_resumes(resumes, job_description):
    results = [(file_name, screen_resume(resume_text, job_description)) for resume_text, file_name in resumes]
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
