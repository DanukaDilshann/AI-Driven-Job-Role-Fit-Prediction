from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import re
import google.generativeai as genai
import pdfplumber
from DB_conn import store_Job_description
import pandas as pd
import PyPDF2 as pdf
import json


load_dotenv()

genai.configure(api_key=os.getenv("google_api_key"))
def run():
## Streamlit App
    # st.set_page_config(page_title="JD Data EXtraction")
    st.header("JD Data EXtraction")
    def extract_text_from_pdf(uploaded_file):
    
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n" if page.extract_text() else ""
        return text.strip()

    col1, col2= st.columns(2)

    with col1:
        input_text = st.text_area("Job description: ", key="input")
    with col2:
        upload_jd = st.file_uploader("Upload your JD (PDF)...", type=["pdf"])
        if upload_jd is not None:
            st.write("PDF uploaded Successfully") 
            extracted_text = extract_text_from_pdf(upload_jd)  
            input_text = extracted_text  
        

    submit0=st.button("JD Data Extraction")

    if submit0:

        def get_gemini_response_description(input_jd, prompt):
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([input_jd, prompt])
            return response.text

# Define the structured prompt
        prompt = (
    "Please analyze the following job description and extract the following details in a structured format:\n"
    "1. Job Title\n"
    "2. Required Programming Skills\n"
    "3. Required Soft Skills\n"
    "4. Required Technical Skills\n"
    "5. Required Educational Qualifications\n"
    "6. Professional Qualifications With Years\n"
    "Response should be a simple list.\n"
    "Very Important: When providing Required Years of Experience, only include the years and relevent possition. "
    "Very Important: When extracting 'Required Years of Experience', include only the years followed by relevant context. If multiple values exist, separate them.\n"
    "Very Important: Required Programming Skills, Required Soft Skills, and Required Technical Skills should be provided as a single row of list items.\n"
    "Very Important: All extracted data should be separated using ','. "
    "Sample output:"
    '''Job Title: Data Scientist
       Required Programming Skills:R,SQL \n
       Required Soft Skills:Communication,Problem-Solving,Teamwork \n
       Required Technical Skills and Software:Machine Learning,TensorFlow  \n
       Required Educational Qualifications:Bachelor’s in Computer Science,Master’s in Data Science (Preferred)\n
       Professional Qualifications with Years:AWS Certified Data Analytics (3+ years),Google Professional ML Engineer (2+ years)\n
       Ensure accuracy and consistency in structuring the extracted details.'''
    )


        response = get_gemini_response_description(input_text, prompt)
        st.write(response)
        match1 = re.search(r"Job Title:\s*(.+)", response)
        match2 = re.search(r"Required Programming Skills:\s*(.+)", response)
        match3 = re.search(r"Required Soft Skills:\s*(.+)", response)
        match4 = re.search(r"Required Technical Skills:\s*(.+)", response)
        match5 = re.search(r"Professional Qualifications With Years:\s*(.+)", response)
        match6 = re.search(r"Required Educational Qualifications:\s*(.+)", response)

# Assign values based on whether matches are found
        job_title = match1.group(1) if match1 else None
        required_programming_skills = match2.group(1) if match2 else None
        required_soft_skills = match3.group(1) if match3 else None
        required_technical_skills = match4.group(1) if match4 else None
        required_years_of_experience = match5.group(1) if match5 else None
        required_educational_qualifications = match6.group(1) if match6 else None  # Handle cases where no match is found

        try:
            store_Job_description( job_title,required_programming_skills,required_soft_skills,required_technical_skills,required_years_of_experience,required_educational_qualifications)
        except:
            st.write("JD Store Fail")
    