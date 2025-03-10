# prompts.py

def get_input_prompt3(selected_jd):
    return f'''📌 AI-Powered Employee Suitability Assessment – Structured & Consistent Prompt
💡 **Objective**:
Evaluate an employee’s profile against a specified job role with a structured, human-friendly assessment that is consistent, accurate, and repeatable across different employee cases.

🔹 **Prompt**
"Analyze the provided employee profile and assess their suitability for the given job role. Ensure a highly structured and consistent response with the following sections:

1️⃣ **Employee Profile Summary**
Provide a clear and structured overview of the employee’s background, including:
- Name
- Current Role & Department
- Age
- Total Years of Experience & Years in Current Role
- Education Qualifications & Professional Certifications
- Key Programming, Technical, Software, and Soft Skills
- Performance Metrics (KPI Score, Goals Achieved, Competency Score, etc.)

2️⃣ **Role Suitability Assessment**
Compare the employee’s skills, experience, and qualifications against the job role requirements.
Categorize skills into:
✅ Essential Skills (Must-have for the role – employee possesses these)
🟡 Core Skills (Important but not mandatory – employee has some but not all)
🔴 Missing Skills (Key gaps preventing immediate suitability)

3️⃣ **Transferable Skills & Transition Readiness**
If the employee lacks some requirements, assess their transition potential by:
- Identifying related experiences that could compensate for skill gaps.
- Highlighting previous job responsibilities that align with the new role.
- Evaluating their learning adaptability and growth potential.

4️⃣ **Job Role Suitability**
📌 Here is the **job description** for the target role:
"{selected_jd}" you have to compare the all the requirement with job description  and comment on his/her suitability.
correctly mentioned the whether his or her sutability or not


5️⃣ **Career Development Recommendations**
Provide specific action items to bridge the gap, such as:
🎯 Recommended Certifications & Courses (e.g., "AWS Certified Machine Learning – Specialty")
📌 Suggested Hands-on Projects (e.g., "Build a Flask API for deploying ML models")
🔹 Mentorship & Learning Pathways (e.g., "Shadow a senior engineer for real-world exposure")

6️⃣ **Estimated Readiness Timeline**
If upskilling is required, estimate how long the employee would need (e.g., 6-12 months).

🔥 **Output Format Guidelines**
- Consistent structure for every evaluation.
- No table format – use engaging, structured paragraphs.
- Data-driven and role-specific insights (no generic assessments).
- Professional, yet conversational tone (like a career coach).

🎯 **Example Input (Structured Employee & Job Role Data)**
Employee Profile:

- Name: James Carter
- Department: Data & AI
- Current Role: Data Analyst
- Age: 29
- Experience: 6 years total (4 in current role)
- Education: Bachelor’s in Statistics and Data Science
- Certifications: Google Data Analytics, Power BI Specialist
- Technical Skills: Python, SQL, R, Data Modeling, Visualization
- Software Skills: Power BI, Tableau, SQL Server
- Soft Skills: Problem-solving, communication, teamwork

Performance Metrics:
- KPI Score: 85%
- Goals Achieved: 15/18
- Competency Score: 90%
- Cultural Alignment Score: 88%

🔍 **Target Job Role:** Senior Data Scientist

📌 **Required Skills:** Python, ML, Deep Learning, Cloud Computing, MLOps
📌 **Experience Requirement:** 5+ years in data science
📌 **Performance Expectations:** Ability to lead predictive analytics projects
'''
