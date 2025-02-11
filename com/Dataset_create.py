import pandas as pd
import pyodbc
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")

lematizer = WordNetLemmatizer()

# Database connection details
server = 'DESKTOP-2DSGQFI'
driver = '{ODBC Driver 17 for SQL Server}'
default_database = 'master'
connection_string = f"DRIVER={driver};SERVER={server};DATABASE={default_database};Trusted_Connection=yes"

conn = pyodbc.connect(connection_string, autocommit=True)
cursor = conn.cursor()
print("Connected to SQL Server")


cursor.execute("USE ABC_Company")
print("Using database ABC_Company")


softskill_query = "SELECT Required_Soft_Skills FROM Job_Descriptions"
proskill_query = "SELECT Required_Programming_Skills FROM Job_Descriptions"
techskill_query = "SELECT Required_Technical_Skills FROM Job_Descriptions"
edu="SELECT Required_Educational_Qualifications FROM Job_Descriptions"


cursor.execute(softskill_query)
softskill_results = cursor.fetchall()
softskill_results = [" ".join(row) for row in softskill_results]  

cursor.execute(proskill_query)
proskill_results = cursor.fetchall()
proskill_results = [" ".join(row) for row in proskill_results]  # use to Convert tuples to strings

cursor.execute(techskill_query)
techskill_results = cursor.fetchall()
techskill_results = [" ".join(row) for row in techskill_results]

cursor.execute(edu)
edu_results = cursor.fetchall()
edu_results = [" ".join(row) for row in edu_results]  

cursor.close()
conn.close()

jd = {
    "Technical": techskill_results,
    "Programming": proskill_results,
    "Soft": softskill_results,
    "educ": edu_results
}

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df = pd.read_excel('C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//src//Employee.xlsx', sheet_name="Sheet1")


df_filtered = df[["EmployeeCode", "List of Technical Skills", "Programming & Software Skills", "List of Soft Skills","Education Qualifications"]].copy()



# def compute_similarity(jd_category, emp_skills_column):
#     jd_text = " ".join(jd_category)
    
#     emp_texts = df_filtered[emp_skills_column].fillna("").astype(str).tolist()
    
#     # TF-IDF app
#     vectorizer = TfidfVectorizer()
#     all_texts = [jd_text] + emp_texts
#     tfidf_matrix = vectorizer.fit_transform(all_texts)
#     jd_vector = tfidf_matrix[0] 
#     employee_vectors = tfidf_matrix[1:]  
#     similarities = cosine_similarity(jd_vector, employee_vectors).flatten()
    
#     return similarities


# df_filtered["Technical Score"] = compute_similarity(jd_skills["Technical"], "List of Technical Skills")
# df_filtered["Programming Score"] = compute_similarity(jd_skills["Programming"], "Programming & Software Skills")
# df_filtered["Soft Score"] = compute_similarity(jd_skills["Soft"], "List of Soft Skills")

# df_filtered=df_filtered.drop(columns=["List of Technical Skills", "Programming & Software Skills", "List of Soft Skills"],axis=1)
# df_filtered = df_filtered.sort_values(by="Emp_id", ascending=True)
# df_merge=df_filtered.merge(df,how='left',on="Emp_id")
# df_merge.drop(columns=["List of Technical Skills", "Programming & Software Skills", "List of Soft Skills"],axis=1,inplace=True)
# print(df_filtered.head(5))
# df_merge.to_excel("Merge_data.xlsx",index=False)





from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

def get_sentence_embeddings(sentences, model):
    embeddings = []
    for sentence in sentences:
        embedding = model.encode(sentence)
        embeddings.append(embedding)
    return np.array(embeddings)

def compute_similarity(set1, set2):
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Get BERT embeddings for both sets
    embeddings1 = get_sentence_embeddings(set1, bert_model)
    embeddings2 = get_sentence_embeddings(set2, bert_model)
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    
    # Compute overall similarity as average of max similarity per word in set1
    max_similarities = np.max(similarity_matrix, axis=1)
    overall_similarity = np.mean(max_similarities)
    
    return overall_similarity

# Example Usage
df_filtered["Technical Score_JD"] = df_filtered["List of Technical Skills"].apply(lambda x: compute_similarity(jd["Technical"], [x]))
df_filtered["Programming Score_JD"] = df_filtered["Programming & Software Skills"].apply(lambda x: compute_similarity(jd["Programming"], [x]))
df_filtered["Soft Score_with_JD"] = df_filtered["List of Soft Skills"].apply(lambda x: compute_similarity(jd["Soft"], [x]))
df_filtered["Education_match_Score_with_JD"] = df_filtered["Education Qualifications"].apply(lambda x: compute_similarity(jd["educ"], [x]))



df_filtered=df_filtered.drop(columns=["List of Technical Skills", "Programming & Software Skills", "List of Soft Skills","Education Qualifications"],axis=1)
df_filtered = df_filtered.sort_values(by="EmployeeCode", ascending=True)
df_merge=df_filtered.merge(df,how='left',on="EmployeeCode")
df_merge.drop(columns=["List of Technical Skills", "Programming & Software Skills", "List of Soft Skills","FullName"],axis=1,inplace=True)
print(df_filtered.head(5))
df_merge.to_excel("Merge_data_new.xlsx",index=False)
