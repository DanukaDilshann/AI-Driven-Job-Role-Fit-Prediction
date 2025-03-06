import streamlit as st
import profile_comparison
import JD_dataExtraction
import scoring

# ✅ Set page config at the very beginning
st.set_page_config(page_title="Multi-App System", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
app_selection = st.sidebar.radio("Select Application", ["Home","Scoring System", "Profile Comparison", "JD Data Extraction"])

# Home Page
if app_selection == "Home":
    st.title("Welcome to the Multi-App System")
    st.write("Select an application from the sidebar.")

elif app_selection =="Scoring System":
    scoring.run()
# Run Profile Comparison App
elif app_selection == "Profile Comparison":
    profile_comparison.run()  

# Run JD Data Extraction App
elif app_selection == "JD Data Extraction":
    JD_dataExtraction.run()


