import random
import pandas as pd
from faker import Faker
import os
from google.cloud import aiplatform
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

genai.configure(api_key=os.getenv("google_api_key"))

# Initialize Faker and set seeds
fake = Faker()
Faker.seed(42)
random.seed(42)

# Function to generate text using Google Gemini Pro
def generate_text(prompt: str, max_output_tokens: int = 256):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Use the correct model
    response = model.predict(prompt=prompt, temperature=0.7, max_output_tokens=max_output_tokens)
    return response.text

# Define constants
num_employees = 50

# Generate synthetic data
synthetic_data = []
for i in range(1, num_employees + 1):
    # Generate role dynamically
    prompt_role = "Generate a current job role for a professional in the data science field."
    current_role = generate_text(prompt_role, max_output_tokens=50)

    # Generate education based on the role
    prompt_education = f"Suggest a relevant degree or educational qualification for a {current_role}."
    education = generate_text(prompt_education, max_output_tokens=50)

    # Generate experience
    years_experience = round(random.uniform(1, 15), 2)

    # Generate technical skills
    prompt_technical_skills = f"List technical skills required for a {current_role}."
    technical_skills = generate_text(prompt_technical_skills, max_output_tokens=100)

    # Generate programming skills
    prompt_programming_skills = f"List programming languages and tools commonly used by a {current_role}."
    programming_skills = generate_text(prompt_programming_skills, max_output_tokens=100)

    # Generate enhanced soft skills
    prompt_soft_skills = f"Generate a detailed list of soft skills for a {current_role}, focusing on interpersonal abilities, leadership qualities, and teamwork skills."
    enhanced_soft_skills = generate_text(prompt_soft_skills, max_output_tokens=100)

    # Generate project descriptions
    prompt_project_description = f"Describe a significant project handled by a {current_role}. Highlight the objectives, tools used, and measurable outcomes."
    enhanced_project_description = generate_text(prompt_project_description, max_output_tokens=150)

    # Assign KPI based on experience and role
    if years_experience > 10:
        kpi = round(random.uniform(4.5, 5), 2)
    elif years_experience > 5:
        kpi = round(random.uniform(4, 4.5), 2)
    else:
        kpi = round(random.uniform(3, 4), 2)

    # Generate employee data
    employee = {
        "Emp_id": i,
        "Emp Name": fake.name(),
        "Age": random.randint(30, 55),
        "Gender": random.choice(["Male", "Female"]),
        "Education Qualifications": education,
        "Professional Qualifications With Years": f"{years_experience} years",
        "Date of Joining": fake.date_this_decade(),
        "Years of Experience in Company": years_experience,
        "List of Technical Skills": technical_skills,
        "Programming & Software Skills": programming_skills,
        "List of Soft Skills": enhanced_soft_skills,
        "Handled Projects": enhanced_project_description,
        "Disciplinary Actions (Yes/No)": random.choice(["Yes", "No"]),
        "Type of Disciplinary Action": random.choice(["interdict", "suspensions", "demotion", "written warning", "disciplinary transfer", "No"]),
        "Promoted Before": random.choice(["Yes", "No"]),
        "Employee Rejoined": random.choice(["Yes", "No"]),
        "Current Role": current_role,
        "KPI": kpi,
    }
    synthetic_data.append(employee)

# Create DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# Save to Excel
output_path = "synthetic_data_with_gemini_pro_dynamic.xlsx"
synthetic_df.to_excel(output_path, index=False)

print(f"Synthetic data generated dynamically and saved to {output_path}")
