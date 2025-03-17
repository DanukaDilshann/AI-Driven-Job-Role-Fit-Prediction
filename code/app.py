import streamlit as st
import profile_comparison
import JD_dataExtraction
import scoring

st.set_page_config(page_title="ABC Company", layout="wide")


st.sidebar.title("Navigation")
app_selection = st.sidebar.radio("Select Application", ["Home","Scoring System", "Profile Comparison", "JD Data Extraction"])

if app_selection == "Home":
    st.title("Welcome to ABC Company Career Growth Hub")
    st.write("Select an application from the sidebar.")

elif app_selection =="Scoring System":
    scoring.run()

elif app_selection == "Profile Comparison":
    profile_comparison.run()  


elif app_selection == "JD Data Extraction":
    JD_dataExtraction.run()


