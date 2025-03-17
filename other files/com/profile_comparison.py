from dotenv import load_dotenv
import base64
import io
import streamlit as st
import os
from PIL import Image
import pdf2image 
import pandas as pd
import PyPDF2 as pdf
import json
import pyodbc
import google.generativeai as genai
from prompt import get_input_prompt3 

load_dotenv()

genai.configure(api_key=os.getenv("google_api_key"))

server = 'DESKTOP-2DSGQFI'
driver = '{ODBC Driver 17 for SQL Server}'
default_database = 'master'
connection_string = f"DRIVER={driver};SERVER={server};DATABASE={default_database};Trusted_Connection=yes"
conn = pyodbc.connect(connection_string, autocommit=True)
cursor = conn.cursor()
print("Connected to SQL Server")

cursor.execute("USE ABC_Company")
print("Using database ABC_Company")

def get_jd_data():
    query = "SELECT * FROM JD_Collection"
    return pd.read_sql(query, conn)

df_jd = get_jd_data()
conn.close()

e_df=pd.read_excel('C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//com//Employee8.xlsx')
# ## Streamlit App
# st.set_page_config(page_title="TalentAligner")
def run():
    st.header("TalentAligner")
# col1, col2= st.columns(2)

# with col1:
#     input_text = st.text_area("Job description: ", key="input")
# with col2:
#     upload_jd = st.file_uploader("Upload your JD (PDF)...", type=["pdf"])
#     if upload_jd is not None:
#         st.write("PDF uploaded Successfully") 

    col1, col2,col3= st.columns(3) 
    with col1:
        option1 = st.selectbox(
            'Job Role',
            set(df_jd['possition'].unique())
        )

# Second dropdown for selecting Department
    with col2:
        option2 = st.selectbox(
            'Select Your Department',
            options=e_df['Department'].unique(),  
            key="Deartment_search"
        )

# Filter EmployeeCode based on selected Department
    filtered_employee_codes = e_df[e_df['Department'] == option2]['EmployeeCode'].unique()

# Third dropdown for selecting Employee Code (filtered by selected Department)
    with col3:
        option3 = st.selectbox(
            'Select Your Employee Number',
            options=filtered_employee_codes, 
            key="employee_search"
        )

    submit1=st.button("Compare the profile with Job Role")



    column_mapping = {
    'Education Qualifications': 'education',
    'Professional Qualifications With Years': 'professional_qualifications',
    'List of Technical Skills': 'technical_skills',
    'Programming & Software Skills': 'programming_skills',
    'List of Soft Skills': 'soft_skills',
    'Projects Completed':'Projects_Completed'
    }

    input_prompt4 ="""
I will provide you with some information about a person, including their age,
 education qualifications, professional qualifications, years of experience, technical skills, programming skills, and soft skills. Based on this data, generate a professional description in jason dump format the person's background, expertise, and capabilities. The description should be formal and highlight their suitability for roles in their field.

Here is the information:
- EmployeeCode:{EmployeeCode}
- Education Qualifications: {education}
- Professional Qualifications: {professional_qualifications}
- List of Technical Skills: {technical_skills}
- Programming Skills: {programming_skills}
- Soft Skills: {soft_skills}
- Projects Completed:{Projects_Completed}
Generate a concise but detailed professional and comprehensive Resume based on the above information.

Very Important:The main goal of the project is to compare an employee's resume with a job description.
Very Important:No resume drafts are required; instead, the system should generate a detailed paragraph summarizing the employee's profile.
Very Important:The paragraph should include:
    Qualifications
    Skills
    Work experience
    Achievements
Very Important:The output should be comprehensive and structured to enable effective comparison with the job description.
Very Important:At this stage, no job description is provided; the focus is solely on creating the employee summary.
"""

# input_prompt3 = """
# You are an skilled employee profile and job description maching  tool for idetifing employee lacking skills that specific Job role, scanner with a deep understanding of data science, Data analyst, Big data engineer ,DEVOPS
# and ATS functionality, 
# your task is to identify the employee profile description against the provided job description. 
# when you are matching job description and profile, give the priority of the 
# skills, education qualifications,profetional qualifications, working experinece etc. not only the words.also i need to identify most essential skills 
# that provided job description.

# very Important: According to the provided job description, identify and clearly classify the required skills into Essential Skills, Core Skills, and Other Skills for the specific job role.
# very Important: Provide a detailed list of the missing skills in the employee profile based on the provided job description.
# very Important:  Identify gaps that employees need to address to improve their chances of succeeding in 
# new job role.provide some suggestions to improve that lacking areas.
# """

# input_prompt3 = """
# You are a skilled employee profile and job description matching tool for identifying the skills that an employee lacks for a specific job role. You have a deep understanding of data science, data analysis, big data engineering, DevOps, and ATS functionality. 

# Your task is to compare the employee profile description with the provided job description and identify the gaps in the employee's skills, qualifications, and experience. 
# When matching the job description and profile, prioritize the following aspects:
# - Skills
# - Education qualifications
# - Professional qualifications
# - Working experience

# It is important to not only focus on the words but also on the essence and relevance of the qualifications, experience, and skills.

# **Very Important**:
# 1. According to the provided job description, clearly classify the required skills into:
#    - Essential Skills
#    - Core Skills
#    - Other Skills
# 2. Provide a detailed list of the missing skills in the employee profile based on the provided job description.
# 3. Identify the gaps that the employee needs to address in order to improve their chances of succeeding in the new job role.
# 4. Offer some suggestions on how the employee can improve in these lacking areas to better align with the job requirements.
# 5. as a final step ,direcly provide whether employee suitable or not for that job possition and give the reason of the choice.
# 6. Do not consider the all the features of the employee. use expereince, skills, qulifications and essential scores with JD.
# """


#     input_prompt3='''You are an advanced AI-powered **Employee Profile and Job Description Matching Tool** designed to identify skill gaps and assess an employee’s suitability for a specific job role. You have deep expertise in **data science, data analysis, big data engineering, DevOps, and ATS functionality** and can evaluate employee profiles based on job descriptions efficiently.

# ### **Task**
# Compare the given **employee profile** with the **job description** and determine the gaps in **skills, qualifications, and experience**. Your goal is to provide an in-depth analysis and recommendations for improving the employee’s chances of succeeding in the role.

# ### first set should be include deatails about the Employee. hear's the order:
#      1. EMployee Name And Employee Code
#      2. Working Departmnet ( this is came from departmnet)
#      3. Current Job role
#      4. Age (age should be in dataset)
#      5. progrmming skills 
#       6.  technical sklls 
#      7.education qulifications    
#      important: all of above deatils should be in point form in a table.that should be show as Profile of employee.
# ### **Matching Criteria**
# 1. **Skills** (Technical , Soft, programming and software)
# 2. **Education Qualifications**
# 3. **Professional Certifications**
# 4. **Relevant Work Experience & Project History(Years of Experience in this Company, Experience in Years Previous Positions)**
# 5. **use the performance matrixes also(Number of Goal Assigned, Number of Goals Achieved, Final Score,
#        Goals Score, Competency Score, Cultural Value Scor,
#        Additional Accomplishment Score, Potential Assessment Score)**

# ### **Structured Output**
# 1. **Classify Required Skills**  
#    - Essential Skills (Must-have for the role)  
#    - Core Skills (Highly valuable but not mandatory)  
#    - Other Skills (Nice to have, additional advantages)  

# 2. **Identify Missing Skills & Gaps**  
#    - Clearly list skills, qualifications, or experiences that are absent in the employee’s profile but are required for the job.  

# 3. **Consider Transferable Skills from Related Roles**  
#    - If the employee has **experience in a related role** (e.g., **Data Engineering, Data Analysis, or similar**) for a **Data Science** position, do **not** automatically mark them as **unsuitable**.  
#    - Instead, highlight how their experience **aligns with the job** and identify what **additional skills they may need to acquire**.  
#    - Likewise, if a Data Scientist applies for a **Data Engineering or Data Analyst role**, consider their ability to transition based on their existing skills.

# 4. **Improvement Recommendations**  
#    - Provide **practical** suggestions for bridging gaps, such as relevant courses, certifications, hands-on projects, or mentorship opportunities.

# ### **Very Important**
# - Focus on the **essence and relevance** of skills, not just keyword matching.
# - Ensure the assessment is **objective, structured, and actionable**.
# - **Do not automatically mark employees as unsuitable just because they have a related but different role**—consider **transferable skills** and the feasibility of transitioning.
# - Use a **logical and data-driven approach** to generate development insights.
# - indicate that employee suitable or not the relevent possition.
# - clearly mentioned, that employee suitable or not.

# Enhance employee career growth by offering personalized **skill development plans and career transition insights** based on experience and project history.'''

#     input_prompt3='''📌 AI-Powered Employee Suitability Assessment – Structured & Consistent Prompt
# 💡 Objective:
# Evaluate an employee’s profile against a specified job role with a structured, human-friendly assessment that is consistent, accurate, and repeatable across different employee cases.

# 🔹 Prompt
# "Analyze the provided employee profile and assess their suitability for the given job role. Ensure a highly structured and consistent response with the following sections:

# 1️⃣ Employee Profile Summary
# Provide a clear and structured overview of the employee’s background, including:
# Name
# Current Role & Department
# Age
# Total Years of Experience & Years in Current Role
# Education Qualifications & Professional Certifications
# Key Programming, Technical, Software, and Soft Skills
# Performance Metrics (KPI Score, Goals Achieved, Competency Score, etc.)
# 2️⃣ Role Suitability Assessment
# Compare the employee’s skills, experience, and qualifications against the job role requirements.
# Categorize skills into:
# ✅ Essential Skills (Must-have for the role – employee possesses these)
# 🟡 Core Skills (Important but not mandatory – employee has some but not all)
# 🔴 Missing Skills (Key gaps preventing immediate suitability)
# 3️⃣ Transferable Skills & Transition Readiness
# If the employee lacks some requirements, assess their transition potential by:
# Identifying related experiences that could compensate for skill gaps.
# Highlighting previous job responsibilities that align with the new role.
# Evaluating their learning adaptability and growth potential.

# 5️⃣ Career Development Recommendations
# Provide specific action items to bridge the gap, such as:
# 🎯 Recommended Certifications & Courses (e.g., "AWS Certified Machine Learning – Specialty")
# 📌 Suggested Hands-on Projects (e.g., "Build a Flask API for deploying ML models")
# 🔹 Mentorship & Learning Pathways (e.g., "Shadow a senior engineer for real-world exposure")
# 6️⃣ Estimated Readiness Timeline
# If upskilling is required, estimate how long the employee would need (e.g., 6-12 months).
# 🔥 Output Format Guidelines
# Consistent structure for every evaluation.
# No table format – use engaging, structured paragraphs.
# Data-driven and role-specific insights (no generic assessments).
# Professional, yet conversational tone (like a career coach).


# very Importnat: All the recomendation should be align with Job distription , i have input.

# 🎯 Example Input (Structured Employee & Job Role Data)
# Employee Profile:

# Name: James Carter
# Department: Data & AI
# Current Role: Data Analyst
# Age: 29
# Experience: 6 years total (4 in current role)
# Education: Bachelor’s in Statistics and Data Science
# Certifications: Google Data Analytics, Power BI Specialist
# Technical Skills: Python, SQL, R, Data Modeling, Visualization
# Software Skills: Power BI, Tableau, SQL Server
# Soft Skills: Problem-solving, communication, teamwork
# Performance Metrics:
# KPI Score: 85%
# Goals Achieved: 15/18
# Competency Score: 90%
# Cultural Alignment Score: 88%
# Target Job Role: Senior Data Scientist

# Required Skills: Python, ML, Deep Learning, Cloud Computing, MLOps
# Experience Requirement: 5+ years in data science
# Performance Expectations: Ability to lead predictive analytics projects
# '''




    def get_gemini_response_Description(input,prompt):
        model=genai.GenerativeModel('gemini-1.5-flash')
        response1=model.generate_content([input,prompt])
        return response1.text

    def get_gemini_response1(input,text,prompt):
        model=genai.GenerativeModel('gemini-1.5-flash')
        responsek=model.generate_content([input,text,prompt])
        return responsek.text

    if submit1:
        if submit1 is not None:
            emp_details = e_df[e_df['EmployeeCode'] == option3]
        # emp_details=emp_details[["EmployeeCode","Professional Qualifications With Years","List of Technical Skills",
        #                          "Programming & Software Skills","List of Soft Skills"]]
            jd=df_jd["possition"]==option1
            selected_jd = df_jd.loc[jd, "Details"].iloc[0]
            input_prompt3 = get_input_prompt3(selected_jd)
            emp_details = emp_details.rename(columns=column_mapping)
        
            if not emp_details.empty:
        # Convert the filtered DataFrame to JSON format
                employee_details_json = emp_details.to_json(orient='records', indent=4)
            else:
                st.write(f"No employee Details found with ID")
                employee_details_json = None

            if employee_details_json:
                response11 = get_gemini_response_Description(input_prompt4, employee_details_json)
                if response11 is not None:
                    response2 = get_gemini_response1(response11,selected_jd,input_prompt3)
                    st.subheader("The response is:")
                    st.write(response2)



# def input_pdf_text(file):
#     reader=pdf.PdfReader(file)
#     text=""
#     for page in reader.pages:
#         text+=page.extract_text() or ""
#     return text
# pdftext=input_pdf_text(upload_jd)

# col1,col2=st.columns(2)
# with col1:
#     submit4=st.text_input("Employee_ID : ")
# with col2:
#    submit5=st.button("Description about the Employee")

# submit1=st.button("Tell me about the resume")
# submit2=st.button("how can i Imporove my skills")

# submit3=st.button("Presentage Match")



# if input_text is not None and pdftext is None:
#     response = get_gemini_response_description(input_text, prompt)
#     st.write(response)
# elif input_text is None and pdftext is not None:
#     response = get_gemini_response_description(pdftext, prompt)
#     st.write(response)    
# elif input_text is not None and pdftext is not None:
#     st.write("Input PDF or Text Format cannot both") 
# else:
#     st.write("Insert JD")


    input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
    """

# input_prompt3 = """
# You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science, Data analyst, Big data engineer ,DEVOPS
# and ATS functionality, 
# your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
# the job description. First the output should come as percentage and then keywords missing and last final thoughts.
# """




# if submit4:
#     if submit4 is not None:
#         emp_details = e_df[e_df['Emp_id'] == submit4]
#         emp_details = emp_details.rename(columns=column_mapping)
    
#         if not emp_details.empty:
#         # Convert the filtered DataFrame to JSON format
#             employee_details_json = emp_details.to_json(orient='records', indent=4)
#         else:
#             st.write(f"No employee found with ID {submit4}")
#             employee_details_json = None

#         if employee_details_json:
#             response11 = get_gemini_response_Description(input_prompt4, employee_details_json)
#             st.subheader("The response is:")
#             st.write(response11)
#     else:
#         st.write("Insert the Employee ID")










