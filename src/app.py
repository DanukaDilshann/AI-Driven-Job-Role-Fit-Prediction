from dotenv import load_dotenv
import base64
import io
import streamlit as st
import os
from PIL import Image
import pdf2image 
import google.generativeai as genai
from Emp_Basic import e_df

load_dotenv()

genai.configure(api_key=os.getenv("google_api_key"))
def get_gemini_response(input,pdf_content,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        ## convert pdf2 image
        images=pdf2image.convert_from_bytes(uploaded_file.read())
        first_page=images[0]

        # convert to bytes
        img_byte_arr=io.BytesIO()
        first_page.save(img_byte_arr,format='JPEG')
        img_byte_arr=img_byte_arr.getvalue()


        pdf_parts=[
            {
                "mime_type":"image/jpeg",
                "data":base64.b64encode(img_byte_arr).decode()  # encode to base64
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")
    


## Streamlit App
st.set_page_config(page_title="TalentAligner")
st.header("TalentAligner")
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("Job description: ", key="input")
with col2:
    upload_jd = st.file_uploader("Upload your JD (PDF)...", type=["pdf"])
    if upload_jd is not None:
        st.write("PDF uploaded Successfully") ## need to create 

uploaded_file=st.file_uploader("Uploader your resume(pdf).........",type=["pdf"])

if uploaded_file is not None:
    st.write("PDF uploaded Successfully")


col1,col2=st.columns(2)
with col1:
    submit4=st.text_input("Employee_ID : ")
with col2:
    st.text("If You are currently employee in this company You can Enter your Employee ID to generate basic description base on your resume")

submit1=st.button("Tell me about the resume")
submit2=st.button("how can i Imporove my skills")
submit3=st.button("Presentage Match")


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


input_prompt3 = """
You are an skilled profile and job description similarity messuring tool, scanner with a deep understanding of data science, Data analyst, Big data engineer ,DEVOPS
and ATS functionality, 
your task is to evaluate the employee profile description against the provided job description. give me the percentage of match if the e,ployee profile matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.also you should provide 
what similarity technique use to meassure the similarity.
"""

input_prompt4 ="""
I will provide you with some information about a person, including their age, education qualifications, professional qualifications, years of experience, technical skills, programming skills, and soft skills. Based on this data, generate a professional description summarizing the person's background, expertise, and capabilities. The description should be formal and highlight their suitability for roles in their field.

Here is the information:
- Name:{name}
- Age: {age}
- Education Qualifications: {education}
- Professional Qualifications: {professional_qualifications}
- List of Technical Skills: {technical_skills}
- Programming Skills: {programming_skills}
- Soft Skills: {soft_skills}
Generate a concise but detailed professional summary based on the above information."""

if submit1:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("the response is ")
        st.write(response)

    else:
        st.write("Please upload the resume")


# if submit3:
#     if uploaded_file is not None:
#         pdf_content=input_pdf_setup(uploaded_file)
#         response=get_gemini_response(input_prompt3,pdf_content,input_text)
#         st.subheader("the response is ")
#         st.write(response)

#     else:
#         st.write("Please upload the resume")



# make employee description
def get_gemini_response_Description(input,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response1=model.generate_content([input,prompt])
    return response1.text

column_mapping = {
    'Emp Name': 'name',
    'Age': 'age',
    'Education Qualifications': 'education',
    'Professional Qualifications With Years': 'professional_qualifications',
    'List of Technical Skills': 'technical_skills',
    'Programming & Software Skills': 'programming_skills',
    'List of Soft Skills': 'soft_skills',
}


if submit4:
    if submit4 is not None:
        emp_details = e_df[e_df['Emp_id'] == submit4]
        emp_details = emp_details.rename(columns=column_mapping)
        
        if not emp_details.empty:
            # Convert the filtered DataFrame to JSON format
            employee_details_json = emp_details.to_json(orient='records', indent=4)
        else:
            st.write(f"No employee found with ID {submit4}")
            employee_details_json = None

        if employee_details_json:
            response11 = get_gemini_response_Description(input_prompt4, employee_details_json)
            st.subheader("The response is:")
            st.write(response11)
    else:
        st.write("Insert the Employee ID")
        

# Pass response1 to the next block

def get_gemini_response1(input,text,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    responsek=model.generate_content([input,text,prompt])
    return responsek.text
if submit3:
    if response11:
        response2 = get_gemini_response1(input_prompt3, response11, input_text)
        st.subheader("The response is:")
        st.write(response2)
    else:
        st.write("Please upload a valid response.")
