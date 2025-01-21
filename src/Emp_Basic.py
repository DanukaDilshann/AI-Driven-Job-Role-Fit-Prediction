

import pandas as pd


df=pd.read_excel("C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//src//k.xlsx")

df['Emp_id'] = df['Emp_id'].astype("string")


e_df=df[['Emp_id','Emp Name', 'Age', 'Gender', 'Education Qualifications',
         'Professional Qualifications With Years','List of Technical Skills',
         'Programming & Software Skills', 'List of Soft Skills']]
# employee_details = e_df[e_df['Emp_id'] == emp_id]
# if not employee_details.empty:
#     print(employee_details)
# else:
#     print(f"No employee found")





# if submit4 is not None:
#     employee_details = e_df[e_df['Emp_id'] == submit4]
# else:
#     print("Enter Your Employee ID")






# if submit4:
#     if submit4.strip():  # Ensure the Employee ID field is not empty or blank
#         emp_details = e_df[e_df['Emp_id'] == submit4]
#         if not emp_details.empty:
#             name=emp_details['Emp Name'].values[0]
#             age = emp_details['Age'].values[0]
#             education = emp_details['Education Qualifications'].values[0]
#             professional_qualifications = emp_details['Professional Qualifications With Years'].values[0]
#             technical_skills = emp_details['List of Technical Skills'].values[0]
#             programming_skills = emp_details['Programming & Software Skills'].values[0]
#             soft_skills = emp_details['List of Soft Skills'].values[0]
#             # Convert the filtered DataFrame to JSON format
#             # employee_details_json = emp_details.to_json(orient='records', indent=4)
#             input_prompt4 ="""
# I will provide you with some information about a person, including their age, education qualifications, professional qualifications, years of experience, technical skills, programming skills, and soft skills. Based on this data, generate a professional description summarizing the person's background, expertise, and capabilities. The description should be formal and highlight their suitability for roles in their field.

# Here is the information:
# - Name:{name}
# - Age: {age}
# - Education Qualifications: {education}
# - Professional Qualifications: {professional_qualifications}
# - List of Technical Skills: {technical_skills}
# - Programming Skills: {programming_skills}
# - Soft Skills: {soft_skills}


# Generate a concise but detailed professional summary based on the above information.

# """
#             response = get_gemini_response_Description(input_prompt4.format(details=emp_details), None)
#             st.subheader("The response is:")
#             st.write(response)
#         else:
#             # Handle the case where no employee is found
#             st.warning(f"No employee found with ID {submit4}")
#     else:
#         st.warning("Insert the Employee ID")

