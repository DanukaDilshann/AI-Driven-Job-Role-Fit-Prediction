import streamlit as st
import pandas as pd

# Load the dataset (replace with actual path or provide user upload option)
def load_data():
    file_path = "C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//com//Score.xlsx"
    file_path1 = "C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//com//Employee8.xlsx"
    df = pd.read_excel(file_path)
    df1 = pd.read_excel(file_path1)
    df1 = df1[df1["Department"] == "Data and AI"]
    return df[['EmployeeCode', 'Suitability_score_scaled']], df1[['EmployeeCode', 'FullName', 'Gender', 'Age', 'Department', 'JobCategory', 'ProficiencyLevel', 'Education Qualifications', 'Years of Experience in this Company', 'Projects Completed']]

# Load employee data
df, df1 = load_data()

# Streamlit app UI
st.title("Employee Score Viewer")

# Dropdown to select Employee Code
selected_employee = st.selectbox("Select Employee Code", df['EmployeeCode'].unique())

# Button to show the score and employee details
if st.button("Show Score"):
    score = round(df[df['EmployeeCode'] == selected_employee]['Suitability_score_scaled'].values[0], 2)
    st.write(f"### Employee Suitability: {score}")
    
    employee_details = df1[df1['EmployeeCode'] == selected_employee]
    if not employee_details.empty:
        st.write("### Employee Details")
        for index, row in employee_details.iterrows():
            st.write(f"- **Full Name:** {row['FullName']}")
            st.write(f"- **Gender:** {row['Gender']}")
            st.write(f"- **Age:** {row['Age']}")
            st.write(f"- **Department:** {row['Department']}")
            st.write(f"- **Job Category:** {row['JobCategory']}")
            st.write(f"- **Proficiency Level:** {row['ProficiencyLevel']}")
            st.write(f"- **Education Qualifications:** {row['Education Qualifications']}")
            st.write(f"- **Years of Experience in this Company:** {row['Years of Experience in this Company']}")
            st.write(f"- **Projects Completed:** {row['Projects Completed']}")
