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

e_df=pd.read_excel('C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//src//Employee.xlsx')
## Streamlit App
st.set_page_config(page_title="TalentAligner")
st.header("Skill Gap Finder")
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
}

input_prompt4 ="""
I will provide you with some information about a person, including their age,
 education qualifications, professional qualifications, years of experience, technical skills, programming skills, and soft skills. Based on this data, generate a professional description summarizing the person's background, expertise, and capabilities. The description should be formal and highlight their suitability for roles in their field.

Here is the information:
- EmployeeCode:{EmployeeCode}
- Education Qualifications: {education}
- Professional Qualifications: {professional_qualifications}
- List of Technical Skills: {technical_skills}
- Programming Skills: {programming_skills}
- Soft Skills: {soft_skills}
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

input_prompt3 = """
You are a skilled employee profile and job description matching tool for identifying the skills that an employee lacks for a specific job role. You have a deep understanding of data science, data analysis, big data engineering, DevOps, and ATS functionality. 

Your task is to compare the employee profile description with the provided job description and identify the gaps in the employee's skills, qualifications, and experience. 
When matching the job description and profile, prioritize the following aspects:
- Skills
- Education qualifications
- Professional qualifications
- Working experience

It is important to not only focus on the words but also on the essence and relevance of the qualifications, experience, and skills.

**Very Important**:
1. According to the provided job description, clearly classify the required skills into:
   - Essential Skills
   - Core Skills
   - Other Skills
2. Provide a detailed list of the missing skills in the employee profile based on the provided job description.
3. Identify the gaps that the employee needs to address in order to improve their chances of succeeding in the new job role.
4. Offer some suggestions on how the employee can improve in these lacking areas to better align with the job requirements.
5. as a final step ,provide that employee suitable or not for that job possition.
"""

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
        emp_details=emp_details[["EmployeeCode","Professional Qualifications With Years","List of Technical Skills",
                                 "Programming & Software Skills","List of Soft Skills"]]
        jd=df_jd["possition"]==option1
        selected_jd = df_jd.loc[jd, "Details"].iloc[0]
       
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










