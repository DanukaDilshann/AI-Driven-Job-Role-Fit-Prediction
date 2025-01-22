import pandas as pd


df=pd.read_excel("C://Users//Danuka//Desktop//ATS//src//k.xlsx")

df['Emp_id'] = df['Emp_id'].astype("string")


e_df=df[['Emp_id','Emp Name', 'Age', 'Gender', 'Education Qualifications',
         'Professional Qualifications With Years','List of Technical Skills',
         'Programming & Software Skills', 'List of Soft Skills']]
# employee_details = e_df[e_df['Emp_id'] == emp_id]
# if not employee_details.empty:
#     print(employee_details)
# else:
#     print(f"No employee found")
