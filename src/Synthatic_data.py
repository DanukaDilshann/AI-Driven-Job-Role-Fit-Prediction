import random
import pandas as pd
from faker import Faker
import os
import requests

# Initialize Faker and set seed for reproducibility
fake = Faker()
Faker.seed(42)
random.seed(42)

# Initialize the text generation model using environment key
def gemini_pipeline(prompt, max_length=100):
    # Simulating model interaction via an API or SDK using an environment key (replace with actual implementation)
    api_key = os.getenv("API_KEY")  # Ensure your API key is set in the environment
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "text-davinci-003", "prompt": prompt, "max_tokens": max_length}
    response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)
    response.raise_for_status()  # Raise an error if the request fails
    return response.json()["choices"][0]["text"].strip()

# Define constants
num_employees = 50

# Generate synthetic data intelligently
synthetic_data = []
for i in range(1, num_employees + 1):
    # Generate role dynamically
    prompt_role = "Generate a current job role for a professional in the data science field."
    current_role = gemini_pipeline(prompt_role, max_length=50)

    # Generate education based on the role
    prompt_education = f"Suggest a relevant degree or educational qualification for a {current_role}."
    education = gemini_pipeline(prompt_education, max_length=50)

    # Generate experience
    years_experience = round(random.uniform(1, 15), 2)

    # Generate technical skills
    prompt_technical_skills = f"List technical skills required for a {current_role}."
    technical_skills = gemini_pipeline(prompt_technical_skills, max_length=100)

    # Generate programming skills
    prompt_programming_skills = f"List programming languages and tools commonly used by a {current_role}."
    programming_skills = gemini_pipeline(prompt_programming_skills, max_length=100)

    # Generate enhanced soft skills
    prompt_soft_skills = f"Generate a detailed list of soft skills for a {current_role}, focusing on interpersonal abilities, leadership qualities, and teamwork skills."
    enhanced_soft_skills = gemini_pipeline(prompt_soft_skills, max_length=100)

    # Generate project descriptions
    prompt_project_description = f"Describe a significant project handled by a {current_role}. Highlight the objectives, tools used, and measurable outcomes."
    enhanced_project_description = gemini_pipeline(prompt_project_description, max_length=150)

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

print(f"Synthetic data generated dynamically with Gemini Pro enhancements and saved to {output_path}")
