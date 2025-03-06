import streamlit as st
import pandas as pd

# Function to load data with caching for efficiency
@st.cache_data
def load_data():
    try:
        # Define file paths
        file_path = "C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//code//Dataset//PCA_scores.xlsx"
        file_path1 = "C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//code//Dataset//Employee8.xlsx"
        file_path2 = "C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//code//Dataset//Autoencoder_score.xlsx"

        # Load datasets
        df_pca = pd.read_excel(file_path)
        df_ae = pd.read_excel(file_path2)
        df_emp = pd.read_excel(file_path1)

        # Filter employees from the "Data and AI" department
        df_emp = df_emp[df_emp["Department"] == "Data and AI"]

        # Select relevant columns
        df_pca = df_pca[['EmployeeCode', 'Suitability_score_scaled_PCA']]
        df_ae = df_ae[['EmployeeCode', 'SuitabilityScore_AE_scaled']]
        df_emp = df_emp[['EmployeeCode', 'FullName', 'Gender', 'Age', 'Department', 'JobCategory', 
                         'ProficiencyLevel', 'Education Qualifications', 'Years of Experience in this Company', 'Projects Completed']]
        
        return df_pca, df_ae, df_emp
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load employee data
df_pca, df_ae, df_emp = load_data()
def run():
# Streamlit UI
    st.title("Employee Suitability Score Viewer")

    if df_pca is not None and df_ae is not None and df_emp is not None:
    # Dropdown to select Employee Code
        selected_employee = st.selectbox("Select Employee Code", df_pca['EmployeeCode'].unique())

    # Button to show the score and employee details
        if st.button("Show Score"):
        # Fetch PCA score
            score_pca = df_pca[df_pca['EmployeeCode'] == selected_employee]['Suitability_score_scaled_PCA']
            score_ae = df_ae[df_ae['EmployeeCode'] == selected_employee]['SuitabilityScore_AE_scaled']

        # Check if values exist
            if not score_pca.empty:
                st.markdown(f"### 🏆 **Suitability Score (PCA):** `{round(score_pca.values[0], 2)}`")
            else:
                st.warning("PCA Score not found for the selected employee.")

            if not score_ae.empty:
                st.markdown(f"### 🏆 **Suitability Score (AE):** `{round(score_ae.values[0], 2)}`")
            else:
                st.warning("Autoencoder Score not found for the selected employee.")

        # Fetch employee details
            employee_details = df_emp[df_emp['EmployeeCode'] == selected_employee]
            if not employee_details.empty:
                st.markdown("### 👤 **Employee Details**")
                for _, row in employee_details.iterrows():
                    st.markdown(f"- **Full Name:** {row['FullName']}")
                    st.markdown(f"- **Gender:** {row['Gender']}")
                    st.markdown(f"- **Age:** {row['Age']}")
                    st.markdown(f"- **Department:** {row['Department']}")
                    st.markdown(f"- **Job Category:** {row['JobCategory']}")
                    st.markdown(f"- **Proficiency Level:** {row['ProficiencyLevel']}")
                    st.markdown(f"- **Education Qualifications:** {row['Education Qualifications']}")
                    st.markdown(f"- **Years of Experience in this Company:** {row['Years of Experience in this Company']}")
                    st.markdown(f"- **Projects Completed:** {row['Projects Completed']}")
            else:
                st.warning("Employee details not found.")
    else:
        st.error("Data not available. Please check file paths or format.")
