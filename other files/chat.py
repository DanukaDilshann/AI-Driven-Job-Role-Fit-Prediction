import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords and lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Sample CV and job description
cv_text = "Experienced Data Scientist skilled in Python, ML, and NLP."
job_text = "We are looking for a Data Scientist with expertise in Python, ML, and natural language processing."

# Preprocess texts
cv_tokens = preprocess_text(cv_text)
job_tokens = preprocess_text(job_text)

# Train a Word2Vec model (use your own pre-trained model for better results)
model = Word2Vec([cv_tokens, job_tokens], vector_size=100, window=5, min_count=1, workers=4)

# Function to calculate mean embedding
def get_mean_embedding(tokens, model):
    embeddings = [model.wv[word] for word in tokens if word in model.wv]
    if embeddings:  # If there are valid embeddings
        return np.mean(embeddings, axis=0)
    else:  # Return a zero vector if no words are in the model
        return np.zeros(model.vector_size)

# Get mean embeddings for CV and job description
cv_embedding = get_mean_embedding(cv_tokens, model)
job_embedding = get_mean_embedding(job_tokens, model)

# Calculate cosine similarity
similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
print(f"Similarity Score: {similarity_score}")

# #############################################################
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight sentence embedding model

# Sample CV and job description
cv_text = "Experienced Data Scientist skilled in Python, machine learning, and NLP."
job_text = "Data Scientist with expertise in Python, ML, and natural language processing."

# Get embeddings for the entire sentences
cv_embedding = model.encode(cv_text)
job_embedding = model.encode(job_text)

# Calculate cosine similarity
similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
print(f"Similarity Score: {similarity_score}")



######################################################3333333333
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Load a pre-trained model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence embedding model

# # Sample CV and job description
# cv_text = "Experienced Data Scientist skilled in Python, machine learning, and NLP."
# job_text = "We are looking for a Data Scientist with expertise in Python, ML, and natural language processing."

# # Tokenize and get embeddings for each token in the text
# def get_mean_embedding(text, model):
#     # Tokenize and get embeddings for each word or token
#     token_embeddings = model.encode(text, output_value='token_embeddings')  # Token-level embeddings
#     if len(token_embeddings) > 0:
#         return np.mean(token_embeddings, axis=0)  # Mean embedding
#     else:
#         return np.zeros(model.get_sentence_embedding_dimension())  # Zero vector if no tokens

# # Get mean embeddings for CV and job description
# cv_embedding = get_mean_embedding(cv_text, model)
# job_embedding = get_mean_embedding(job_text, model)

# # Calculate cosine similarity
# similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
# print(f"mean Similarity Score: {similarity_score}")



import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample CV and job description
cv_text = "Experienced Data Scientist skilled in Python, machine learning, and NLP."
job_text = "We are looking for a Data Scientist with expertise in Python, ML, and natural language processing."

# Tokenize and get embeddings for each token in the text
def get_mean_embedding(text, model):
    # Tokenize and get embeddings for each word or token
    token_embeddings = model.encode(text, output_value='token_embeddings')  # Token-level embeddings

    # Convert to NumPy array if necessary and calculate the mean
    if len(token_embeddings) > 0:
        return np.mean(token_embeddings.numpy(), axis=0)  # Convert PyTorch tensor to NumPy array
    else:
        return np.zeros(model.get_sentence_embedding_dimension())  # Zero vector if no tokens

# Get mean embeddings for CV and job description
cv_embedding = get_mean_embedding(cv_text, model)
job_embedding = get_mean_embedding(job_text, model)

# Calculate cosine similarity
similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
print(f"Similarity Score: {similarity_score}")

