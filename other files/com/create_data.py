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

df = pd.read_excel('C://Users//DanukaDilshanRathnay//Desktop//AI-Driven-Job-Role-Fit-Prediction//com//Employee6.xlsx', sheet_name="Sheet1")

df_filtered = df[["EmployeeCode", "List of Technical Skills", "List of Programming Skills", "List of Soft Skills","Education Qualifications"]].copy()

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
    
    if not set1 or not set2 or set1 == [""] or set2 == [""]:
        return 0.0 
    embeddings1 = get_sentence_embeddings(set1, bert_model)
    embeddings2 = get_sentence_embeddings(set2, bert_model)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    max_similarities = np.max(similarity_matrix, axis=1)
    overall_similarity = np.mean(max_similarities)
    
    return overall_similarity

# Ensure no NaN values before applying similarity computation  # CHANGED: Added .fillna("").astype(str) to prevent NaN errors
df_filtered["Technical Score_JD"] = df_filtered["List of Technical Skills"].fillna("").astype(str).apply(lambda x: compute_similarity(jd["Technical"], [x]))
df_filtered["Programming Score_JD"] = df_filtered["List of Programming Skills"].fillna("").astype(str).apply(lambda x: compute_similarity(jd["Programming"], [x]))
df_filtered["Soft Score_with_JD"] = df_filtered["List of Soft Skills"].fillna("").astype(str).apply(lambda x: compute_similarity(jd["Soft"], [x]))
df_filtered["Education_match_Score_with_JD"] = df_filtered["Education Qualifications"].fillna("").astype(str).apply(lambda x: compute_similarity(jd["educ"], [x]))


df_filtered=df_filtered.drop(columns=["List of Technical Skills", "List of Programming Skills", "List of Soft Skills","Education Qualifications"],axis=1)
df_filtered = df_filtered.sort_values(by="EmployeeCode", ascending=True)
df_filtered.to_excel("score.xlsx",index=False)
df_merge=df_filtered.merge(df,how='left',on="EmployeeCode")
df_merge.drop(columns=["List of Technical Skills", "List of Programming Skills", "List of Soft Skills","FullName"],axis=1,inplace=True)
print(df_filtered.head(5))
df_merge.to_excel("Merge_data_new6.xlsx",index=False)



