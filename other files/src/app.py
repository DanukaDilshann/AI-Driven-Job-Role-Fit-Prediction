from dotenv import load_dotenv
import base64
import io
import streamlit as st
import os
from PIL import Image
import pdf2image 
import google.generativeai as genai
# from Emp_Basic import e_df
import re
import pandas as pd
import PyPDF2 as pdf
import json


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
                "data":base64.b64encode(img_byte_arr).decode()  
            }
        ]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")
import pyodbc
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


# def get_emp_data():
#     query = "SELECT * FROM Employee"
#     return pd.read_sql(query, conn)

# df_emp = get_emp_data()

# Close the connection after fetching data
conn.close()
e_df=pd.read_excel('C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//src//Employee.xlsx')
## Streamlit App
st.set_page_config(page_title="TalentAligner")
st.header("Skill Gap Finder")
col1, col2= st.columns(2)

with col1:
    input_text = st.text_area("Job description: ", key="input")
with col2:
    upload_jd = st.file_uploader("Upload your JD (PDF)...", type=["pdf"])
    if upload_jd is not None:
        st.write("PDF uploaded Successfully") 


col1, col2= st.columns(2) 
with col1:      
    option2 = st.selectbox(
        'Job Role',
        set(df_jd['possition'].unique()))
with col2:
    # input_ID = st.text_area("Employee ID: ", key="input")
    option1 = st.selectbox(
    'Select Your Employee Number',
    options=e_df['EmployeeCode'].unique(),  # You can directly use the unique employee codes here
    key="employee_search"
)

submit1=st.button("Compare the profile with Job Role")

column_mapping = {
    'Emp Name': 'name',
    'Age': 'age',
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
- Name:{name}
- Age: {age}
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

input_prompt3 = """
You are an skilled employee profile and job description similarity messuring tool for selecting the employee in vacant job position, scanner with a deep understanding of data science, Data analyst, Big data engineer ,DEVOPS
and ATS functionality, 
your task is to evaluate the employee profile description against the provided job description. give me the percentage of match if the employee profile matches
the job description. First the output should come as percentage and then keywords missing and last final thoughts.also you should provide 
what similarity technique use to meassure the similarity .when you are matching job description and profile, give the priority of the 
skills, education qualifications,profetional qualifications, working experinece etc. not only the words.also i need to identify most essential skills 
that provided job description and give the maximum weight for it.then measure the similarity very stricly.

very Important: According to the provided job description, identify and clearly classify the required skills into Essential Skills, Core Skills, and Other Skills for the specific job role.
very Important: Provide a detailed list of the missing skills in the employee profile based on the provided job description.
very Important:  Identify gaps that employees need to address to improve their chances of succeeding in 
new job role.
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
        emp_details = e_df[e_df['EmployeeCode'] == option1]
        jd=df_jd["Details"]==option2
        # emp_details = emp_details.rename(columns=column_mapping)
        if not emp_details.empty:
        # Convert the filtered DataFrame to JSON format
            employee_details_json = emp_details.to_json(orient='records', indent=4)
        else:
            st.write(f"No employee Details found with ID")
            employee_details_json = None

        if employee_details_json:
            response11 = get_gemini_response_Description(input_prompt4, employee_details_json)
            if response11 is not None:
                response2 = get_gemini_response1(response11,jd,input_prompt3)
                st.subheader("The response is:")
                st.write(response2)




# submit0=st.button("JD Data Extraction")




def input_pdf_text(file):
    reader=pdf.PdfReader(file)
    text=""
    for page in reader.pages:
        text+=page.extract_text() or ""
    return text
pdftext=input_pdf_text(upload_jd)

col1,col2=st.columns(2)
with col1:
    submit4=st.text_input("Employee_ID : ")
with col2:
   submit5=st.button("Description about the Employee")

# submit1=st.button("Tell me about the resume")
# submit2=st.button("how can i Imporove my skills")

# submit3=st.button("Presentage Match")



if input_text is not None and pdftext is None:
    response = get_gemini_response_description(input_text, prompt)
    st.write(response)
elif input_text is None and pdftext is not None:
    response = get_gemini_response_description(pdftext, prompt)
    st.write(response)    
elif input_text is not None and pdftext is not None:
    st.write("Input PDF or Text Format cannot both") 
else:
    st.write("Insert JD")


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




# # make employee description



column_mapping = {
    'Emp Name': 'name',
    'Age': 'age',
    'Education Qualifications': 'education',
    'Professional Qualifications With Years': 'professional_qualifications',
    'List of Technical Skills': 'technical_skills',
    'Programming & Software Skills': 'programming_skills',
    'List of Soft Skills': 'soft_skills',
}


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


# def get_gemini_response1(input,text,prompt):
#     model=genai.GenerativeModel('gemini-1.5-flash')
#     responsek=model.generate_content([input,text,prompt])
#     return responsek.text


# def extract_similarity_score(response_text):
#     from Similarity import similarity_score11
#     st.write(f"Employee_ID: {submit4} from Similarity Score: {similarity_score}%")
#     match = re.search(r'(//d+//.?//d*)//s*%', response_text)
#     if match:
#         return float(match.group(1))
#     return None


# # At the end of your submit3 logic:
# if submit3:
#     if response11:
#         response2 = get_gemini_response1(input_prompt3, response11, input_text)
#         st.subheader("The response is:")
#         st.write(response2)

#         similarity_scoreq = extract_similarity_score(response2)
#         if similarity_scoreq and submit4 is not None:
#             # Save the result to SQL database
#             store_similarity_score(int(submit4), similarity_scoreq)
#             st.write(f"Employee_ID: {submit4} stored with Similarity Score: {similarity_scoreq}%")

        
#         else:
#             st.write("Could not extract similarity score.")
#     else:
#         st.write("Please upload a valid response.")







# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Download stopwords and lemmatizer
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Preprocessing function
# def preprocess_text(text):
#     # Lowercase the text
#     text = text.lower()
#     # Remove special characters and digits
#     text = re.sub(r'[^a-z//s]', '', text)
#     # Tokenize
#     tokens = text.split()
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     # Lemmatize
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return tokens


# # Load a pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight sentence embedding model

# if submit3:
#     if response11 and input_text is not None:
# # Preprocess the input text
#         cv_text = preprocess_text(response11)
#         job_text = preprocess_text(input_text)

# # Convert the tokenized text back to a single string (to be processed by the SentenceTransformer)
#         cv_text_string = " ".join(cv_text)
#         job_text_string = " ".join(job_text)

# # Get embeddings for the entire sentences
#         cv_embedding = model.encode(cv_text_string)
#         job_embedding = model.encode(job_text_string)

# # # Check the shape of embeddings
# # print("CV Embedding Shape:", len(cv_embedding))
# # print("Job Embedding Shape:", len(job_embedding))

# # Now calculate cosine similarity
#         similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
#         st.write(f"Similarity Score: {similarity_score}")
#         if isinstance(similarity_score,bytes):
#             similarity_score=similarity_score.decode("utf-8")
#             match = re.search(r"Similarity Score: (//d+//.?//d*)", similarity_score)
#             if match:
#                 similarity_score = float(match.group(1))
#                 formatted_score = f"{similarity_score:.2f}"
#                 df=pd.DataFrame({
#                       "Emp_id" : [] ,
#                     "Similarity_Score" : []
#                 })
#                 new_row = {'Emp_id': submit4 , 'Similarity_Score': formatted_score}
#                 df = df.append(new_row, ignore_index=True)
#                 df.to_excel("second.xlsx",index=False)
#             else:
#                 print("Similarity score not found in the text.")
#     else:
#         st.write("Check EMP profile or JD ")



# def get_gemini_response1(input,text,prompt):
#     model=genai.GenerativeModel('gemini-1.5-flash')
#     responsek=model.generate_content([input,text,prompt])
#     return responsek.text




# import pandas as pd
# # from genai.model import GenerativeModel  # Ensure this is the correct import for the genai library

# def extract_info_from_jd(jd_text):
#     """Extract job-related information from a job description using eGemini."""
#     # Prompt for the eGemini model
#     prompt = f"""
#     Extract the following information from the job description:
#     1. Job Title
#     2. Programming Skills
#     3. Required Soft Skills
#     4. Required Technical Skills
#     5. Required Years of Experience
#     6. Required Educational Qualifications

#     Job Description:
#     {jd_text}

#     Output the results in JSON format.
#     """
#     try:
#         # Initialize the model
#         model = GenerativeModel("gemini-1.5-flash")  # Ensure the model name is correct
#         response = model.generate_content(prompt=prompt)
        
#         # Parse the model's response
#         extracted_data = eval(response)  # Convert JSON-like string to Python dict
#         return extracted_data
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# def save_to_excel(data, filename="job_details.xlsx"):
#     """Save extracted data to an Excel file."""
#     df = pd.DataFrame(data)
#     df.to_excel(filename, index=False)
#     print(f"Data saved to {filename}")

# # Sample job descriptions (you can replace this with dynamic input or a list)
# job_descriptions =input_text
# # Extracted information storage
# extracted_info = []

# for jd in job_descriptions:
#     result = extract_info_from_jd(jd)
#     if result:
#         extracted_info.append(result)

# # Save the extracted information to an Excel file
# if extracted_info:
#     save_to_excel(extracted_info)
# df=pd.read_excel("job_details.xlsx")
# st.write(df.head(1))
