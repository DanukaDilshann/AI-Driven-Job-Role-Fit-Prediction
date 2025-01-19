from dotenv import load_dotenv
import base64
load_dotenv()
import io
import streamlit as st
import os
from PIL import Image
import pdf2image 
import google.generativeai as genai

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
st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS tracking system")
input_text=st.text_area("Job description: ",key="input")
uploaded_file=st.file_uploader("Uploader your resume(pdf).........",type=["pdf"])


if uploaded_file is not None:
    st.write("PDF uploaded Successfully")
    
submit1=st.button("Tell me about the resume")
submit2=st.button("how can i Imporove my skills")
submit3=st.button("Presentage Match")

input_prompt1 = """
 You are an experienced Technical Human Resource Manager,your task is to review the provided resume against the job description. 
  Please share your professional evaluation on whether the candidate's profile aligns with the role. 
 Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science, Data analyst, Big data engineer ,DEVOPS
and ATS functionality, 
your task is to evaluate the resume against the provided job description. give me the percentage of match if the resume matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.
"""


if submit1:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt1,pdf_content,input_text)
        st.subheader("the response is ")
        st.write(response)

    else:
        st.write("Please upload the resume")


if submit3:
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        response=get_gemini_response(input_prompt3,pdf_content,input_text)
        st.subheader("the response is ")
        st.write(response)

    else:
        st.write("Please upload the resume")
