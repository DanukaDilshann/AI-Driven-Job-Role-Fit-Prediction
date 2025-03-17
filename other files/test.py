# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Example JD and skills
# jd_text = """OXO Tech is a specialist in technical recruitment, connecting skilled professionals with leading Independent Software Vendors (ISVs) globally. A prominent US-based software development company is urgently seeking resources for above positions at their Colombo-based development center

# Responsibilities


# """

# skills = ["Python", "sql", "machine_learning", "excel","php"]

# # Generate embeddings
# jd_embedding = model.encode([jd_text])[0]
# skill_embeddings = model.encode(skills)

# # Compute cosine similarity
# from sklearn.metrics.pairwise import cosine_similarity

# similarities = cosine_similarity([jd_embedding], skill_embeddings)[0]

# weights = {skill: similarities[i] for i, skill in enumerate(skills)}
# rounded_weights = {skill: round(similarity, 2) for skill, similarity in weights.items()}
# print(rounded_weights)




# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# # Sample JD with categorized skills
# jd_skills = {
#     "Technical": ["Machine Learning", "Data Analysis", "Statistics", "Big Data"],
#     "Programming": ["Python", "SQL", "R"],
#     "Soft": ["Communication", "Leadership", "Problem-Solving"]
# }

# # Employee skills categorized
# employee_skills = {
#     "Emp1": {
#         "Technical": ["Data Analysis", "Statistics", "Data Mining"],
#         "Programming": ["Python", "Java"],
#         "Soft": ["Leadership", "Communication"]
#     },
#     "Emp2": {
#         "Technical": ["Machine Learning", "Big Data", "AI"],
#         "Programming": ["SQL", "R", "Python"],
#         "Soft": ["Teamwork", "Problem-Solving"]
#     },
#     "Emp3": {
#         "Technical": ["Cloud Computing", "Data Warehousing"],
#         "Programming": ["C++", "JavaScript"],
#         "Soft": ["Creativity", "Adaptability"]
#     }
# }

# # Define function to calculate TF-IDF similarity
# def compute_similarity(jd_category, emp_category):
#     # Convert lists to strings
#     jd_text = " ".join(jd_category)
#     emp_texts = [" ".join(emp_category) for emp_category in emp_category.values()]
    
#     # TF-IDF Vectorization
#     vectorizer = TfidfVectorizer()
#     all_texts = [jd_text] + emp_texts
#     tfidf_matrix = vectorizer.fit_transform(all_texts)

#     # Compute Cosine Similarity (JD vs Employees)
#     jd_vector = tfidf_matrix[0]  # JD is first row
#     employee_vectors = tfidf_matrix[1:]  # Employees are remaining rows
#     similarities = cosine_similarity(jd_vector, employee_vectors).flatten()
    
#     return similarities

# # Compute similarity for each category
# technical_scores = compute_similarity(jd_skills["Technical"], {emp: skills["Technical"] for emp, skills in employee_skills.items()})
# programming_scores = compute_similarity(jd_skills["Programming"], {emp: skills["Programming"] for emp, skills in employee_skills.items()})
# soft_scores = compute_similarity(jd_skills["Soft"], {emp: skills["Soft"] for emp, skills in employee_skills.items()})

# # Store results
# employee_names = list(employee_skills.keys())
# df = pd.DataFrame({
#     "Employee": employee_names,
#     "Technical Score": technical_scores,
#     "Programming Score": programming_scores,
#     "Soft Score": soft_scores
# })

# # # (Optional) Weighted Final Score (adjust weights if needed)
# # weights = {"Technical": 0.4, "Programming": 0.35, "Soft": 0.25}
# # df["Final Score"] = (df["Technical Score"] * weights["Technical"] +
# #                      df["Programming Score"] * weights["Programming"] +
# #                      df["Soft Score"] * weights["Soft"])

# # # Sort by Final Score
# # df = df.sort_values(by="Final Score", ascending=False)

# print(df)





# import spacy

# nlp = spacy.load("en_core_web_lg")  # Use "en_core_web_lg" for better accuracy

# # Word similarity
# w1="python"
# w2="Python"
# word1 = nlp(w1)
# word2 = nlp(w2)

# print(f"Similarity: {word1.similarity(word2)}")  


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample skill sets
skill_set_1 = "Leadership, Creativity, Teamwork, Adaptability, Critical Thinking"
skill_set_2 = "Problem-solving, Teamwork, Communication (written and verbal)"

# Preprocess skills (convert into list of words)
skills_list = [simple_preprocess(skill_set_1), simple_preprocess(skill_set_2)]

# Tag documents (needed for training)
tagged_data = [TaggedDocument(words=skills, tags=[str(i)]) for i, skills in enumerate(skills_list)]

# Train Doc2Vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs=100)

# Get vector representations
vec1 = model.infer_vector(simple_preprocess(skill_set_1))
vec2 = model.infer_vector(simple_preprocess(skill_set_2))

# Compute cosine similarity
similarity = cosine_similarity([vec1], [vec2])[0][0]
print(f"Skill Set Similarity: {similarity:.2f}")


from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample skill sets
skills_1 = ["Leadership", "Creativity", "Teamwork", "Adaptability", "Critical Thinking"]
skills_2 = ['Problem-solving', 'Teamwork', 'Communication (written and verbal)']

# Convert to sentence format
skills_1_text = " ".join(skills_1)
skills_2_text = " ".join(skills_2)

# Step 1: Compute TF-IDF weights
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([skills_1_text, skills_2_text])
tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
print(vectorizer.vocabulary_)
print(tfidf_matrix.toarray())
# Step 2: Generate BERT embeddings
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def weighted_embedding(skills, tfidf_scores):
    embeddings = []
    weights = []
    
    for skill in skills:
        embedding = bert_model.encode(skill)
        weight = tfidf_scores.get(skill.lower(), 1.0)  # Default weight = 1 if not in TF-IDF
        embeddings.append(embedding * weight)
        weights.append(weight)
        print(weight)
    
    # Compute weighted average of embeddings
    return np.average(embeddings, axis=0, weights=weights)

vec1 = weighted_embedding(skills_1, tfidf_scores)
vec2 = weighted_embedding(skills_2, tfidf_scores)

# Step 3: Compute similarity
similarity = cosine_similarity([vec1], [vec2])[0][0]
print(f"Skill Set Similarity both: {similarity:.2f}")



from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_sentence_embeddings(sentences, model, tokenizer):
    tokens = {'input_ids': [], 'attention_mask': []}
    for sentence in sentences:
        new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention = tokens['attention_mask']
    
    mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
    mask_embeddings = embeddings * mask
    
    summed = torch.sum(mask_embeddings, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    
    mean_pooled = summed / counts
    
    return mean_pooled.detach().numpy()

def compute_similarity(set1, set2):
    model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    embeddings1 = get_sentence_embeddings(set1, model, tokenizer)
    embeddings2 = get_sentence_embeddings(set2, model, tokenizer)
    
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    overall_similarity = np.mean(similarity_matrix)
    
    return overall_similarity, similarity_matrix

# Example Usage
set1 = ["Leadership", "Creativity", "Teamwork", "Adaptability", "Critical Thinking"]
set2 = ['Problem-solving', 'Teamwork', 'Communication (written and verbal)']

overall_similarity, similarity_matrix = compute_similarity(set1, set2)
print("Overall Similarity:", overall_similarity)
print("Similarity Matrix:")
print(similarity_matrix)
