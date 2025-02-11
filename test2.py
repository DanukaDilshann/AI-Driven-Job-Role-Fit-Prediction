from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

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

def weighted_embedding(skills, tfidf_scores, bert_model):
    embeddings = []
    weights = []
    
    for skill in skills:
        embedding = bert_model.encode(skill)
        weight = tfidf_scores.get(skill.lower(), 1.0)  # Default weight = 1 if not in TF-IDF
        embeddings.append(embedding * weight)
        weights.append(weight)
    
    return np.average(embeddings, axis=0, weights=weights)

def compute_similarity(set1, set2):
    model_name = 'sentence-transformers/bert-base-nli-mean-tokens'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Compute TF-IDF weights
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([" ".join(set1), " ".join(set2)])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))
    
    # Compute weighted BERT embeddings
    vec1 = weighted_embedding(set1, tfidf_scores, bert_model)
    vec2 = weighted_embedding(set2, tfidf_scores, bert_model)
    
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity

# Example Usage
set1 = ["Leadership", "Creativity", "Teamwork", "Adaptability", "Critical Thinking"]
set2 = ['Problem-solving', 'Teamwork', 'Communication (written and verbal)']

similarity_score = compute_similarity(set1, set2)
print("Hybrid TF-IDF + BERT Similarity:", similarity_score)



from transformers import AutoTokenizer, AutoModel
import torch
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
    
    return overall_similarity, similarity_matrix

# Example Usage
set1 = ['Machine Learning', 'Deep Learning', 'Data Analysis',' Predictive Modeling', 'Statistical Analysis']

set2 = ['Machine learning libraries (TensorFlow, Keras, scikit-learn), knowledge of advanced statistical techniques (regression, properties of distributions, statistical tests)']

overall_similarity, similarity_matrix = compute_similarity(set1, set2)
print("Overall Similarity last:", overall_similarity)
print("Similarity Matrix last:")
print(similarity_matrix)

