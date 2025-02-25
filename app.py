import streamlit as st
import google.generativeai as genai
import PyPDF2 as pdf
import time

# Configure API key
genai.configure(api_key="AIzaSyB6Xb5Z-zb4evZg81rntCMfxU34srkWX0s")

# Function to instantiate model and get response
def get_gemini_response(input):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(input)
    return response.text

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    return text

# Prompt Template
input_prompt = """
Hey act like a skilled or very experienced ATS (Application Tracking System)
with a deep understanding of tech field, software engineering, data science, data analyst
and bit data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide best assistance
for improving the resumes. Assign the percentage matching based on JD (Job Description)
and the missing keywords with high accuracy.

I want the response in json structure like
{
    "JD Match": "%",
    "Missing Keywords": [],
    "Profile Summary": ""
}
"""

# Streamlit UI
st.set_page_config(page_title="ATS Resume Screening", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>🚀 ATS Resume Screening</h1>", 
    unsafe_allow_html=True
)
st.markdown("<h4 style='text-align: center;'>🔍 Match Your Resume Against Job Description</h4>", unsafe_allow_html=True)

st.divider()
jd = st.text_area("📄 Paste the Job Description", height=150, help="Provide the JD for matching.")
uploaded_file = st.file_uploader("📤 Upload Your Resume (PDF)", type="pdf", help="Upload your resume for screening.")

if st.button("🚀 Analyze Resume", use_container_width=True):
    if uploaded_file:
        text = input_pdf_text(uploaded_file)
        
        st.toast("⏳ Screening in progress... Please wait.", icon="⏳")
        
        response_json = get_gemini_response([input_prompt, f"Job Description:\n{jd}", f"Resume:\n{text}"])
        
        try:
            response_data = eval(response_json)  # Convert string JSON to dictionary
            
            jd_match = int(response_data["JD Match"].replace("%", "").strip())
            missing_keywords = response_data["Missing Keywords"]
            profile_summary = response_data["Profile Summary"]

            st.divider()

            # Horizontal Layout using Columns
            col1, col2, col3 = st.columns([1, 1, 2])  # Adjust width ratios

            # JD Match Percentage
            with col1:
                st.subheader("📊 JD Match")
                st.progress(jd_match / 100)
                st.info(f"✅ Matched: **{jd_match}%**")

            # Missing Keywords
            with col2:
                st.subheader("🔑 Missing Keywords")
                if missing_keywords:
                    st.warning(", ".join(missing_keywords))
                else:
                    st.success("🎉 No missing keywords!")

            # Profile Summary
            with col3:
                st.subheader("📝 Profile Summary")
                st.text_area("Summary", profile_summary, height=150, disabled=True)

        except Exception as e:
            st.error("⚠️ Error processing the response. Please try again.")

    else:
        st.error("⚠️ Please upload your resume before submitting.")
