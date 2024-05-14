import streamlit as st
from openai import OpenAI
from keys import openai_key, mysql_password
import nltk
from nltk.corpus import stopwords
import string
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from elasticsearch import Elasticsearch
import mysql.connector
from mysql.connector import Error

# Initialize the OpenAI client
client = OpenAI(api_key=openai_key)
es = Elasticsearch("http://localhost:9200")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
HOST_NAME = 'localhost'
USER_NAME = 'root'
DATABASE_NAME = 'ARXIVPaperDB'
USER_PASSWORD = mysql_password

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.title('OpenAI Chatbot: ArXiv Paper Database')

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]

def get_response(messages, context, objective, structure, tone, audience, result, additional_context):
    prompt = f"""
    Context: {context}
    Objective: {objective}
    Structure: {structure}
    Tone: {tone}
    Audience: {audience}
    Result: {result}
    Additional Context: {additional_context}
    """
    messages.append({"role": "system", "content": prompt})
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    return response.choices[0].message.content.strip()

def remove_stopwords(text):
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned_words)

def create_mysql_connection():
    try:
        connection = mysql.connector.connect(
            host=HOST_NAME,
            user=USER_NAME,
            password=USER_PASSWORD,
            database=DATABASE_NAME
        )
        return connection
    except Error as e:
        print(f"The error '{e}' occurred")
        return None

def search_documents_in_elasticsearch(query_text, top_k=10):
    try:
        response = es.search(
            index='arxiv_paper_chunks',
            body={
                "query": {
                    "match": {
                        "text": query_text
                    }
                }
            },
            size=top_k
        )

        top_results = [(hit['_source']['chunk_id'], hit['_score']) for hit in response['hits']['hits']]
        return top_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def vectorize_query(query):
    query_vector = model.encode(query, convert_to_tensor=False)
    return query_vector

def get_faiss_index():
    connection = create_mysql_connection()
    cursor = connection.cursor()
    query = "SELECT EMBEDDING, ID FROM CHUNK"
    cursor.execute(query)
    embeddings = []
    chunk_ids = []
    for embedding, chunk_id in cursor:
        np_embedding = np.frombuffer(embedding, dtype=np.float32)
        embeddings.append(np_embedding)
        chunk_ids.append(chunk_id)
    cursor.close()
    connection.close()
    embeddings_array = np.array(embeddings)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index, chunk_ids

def search_chunks_in_faiss(query_vector, index, chunk_ids, top_k=10):
    query_vector = np.array([query_vector])
    distances, indices = index.search(query_vector, top_k)
    top_chunks = [(chunk_ids[idx], dist) for idx, dist in zip(indices[0], distances[0])]
    return top_chunks

def normalize_scores(results, search_type):
    if not results:
        return []

    scores = [result[1] for result in results]
    max_score = max(scores)
    min_score = min(scores)

    if search_type == 'faiss':
        normalized_scores = [(results[i][0], 1 - (scores[i] - min_score) / (max_score - min_score)) for i in range(len(results))]
    elif search_type == 'elastic':
        normalized_scores = [(results[i][0], (scores[i] - min_score) / (max_score - min_score)) for i in range(len(results))]
    else:
        raise ValueError("Invalid score_type. Use 'faiss' or 'elastic'.")

    return normalized_scores

def combine_and_sort_scores(normalized_faiss_scores, normalized_elastic_scores):
    combined_scores = {}
    for chunk_id, score in normalized_faiss_scores:
        combined_scores[chunk_id] = score
    for chunk_id, score in normalized_elastic_scores:
        if chunk_id in combined_scores:
            combined_scores[chunk_id] += score
        else:
            combined_scores[chunk_id] = score

    sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_combined_scores

def extract_ids_and_scores(sorted_scores):
    retrieved_ids = [item[0] for item in sorted_scores]
    final_scores = [item[1] for item in sorted_scores]
    return retrieved_ids, final_scores

def get_chunk_text_by_id(chunk_ids):
    connection = create_mysql_connection()
    cursor = connection.cursor()

    format_strings = ','.join(['%s'] * len(chunk_ids))
    query = f"SELECT TEXT FROM CHUNK WHERE ID IN ({format_strings})"
    cursor.execute(query, tuple(chunk_ids))
    results = cursor.fetchall()
    texts = [result[0] for result in results]

    cursor.close()
    connection.close()

    return texts

def get_doc_id_by_chunk_id(chunk_ids):
    connection = create_mysql_connection()
    cursor = connection.cursor()

    format_strings = ','.join(['%s'] * len(chunk_ids))
    query = f"SELECT DOCUMENT_ID FROM CHUNK WHERE ID IN ({format_strings})"
    cursor.execute(query, tuple(chunk_ids))
    results = cursor.fetchall()
    doc_ids = [result[0] for result in results]

    cursor.close()
    connection.close()

    return doc_ids

def get_metadata_primary_ids(document_ids):
    connection = create_mysql_connection()
    cursor = connection.cursor()

    primary_ids = []
    query = "SELECT ID FROM METADATA WHERE DOCUMENT_ID = %s"

    for document_id in document_ids:
        cursor.execute(query, (document_id,))
        result = cursor.fetchone()
        if result:
            primary_ids.append(result[0])

    cursor.close()
    connection.close()
    return primary_ids

def get_file_name_by_doc_id(document_ids):
    connection = create_mysql_connection()
    cursor = connection.cursor()

    file_names = []
    query = "SELECT FILE_NAME FROM DOCUMENT WHERE ID = %s"

    for document_id in document_ids:
        cursor.execute(query, (document_id,))
        result = cursor.fetchone()
        if result:
            file_names.append(result[0])

    cursor.close()
    connection.close()
    return file_names

def get_summaries_by_metadata_id(metadata_ids):
    connection = create_mysql_connection()
    cursor = connection.cursor()

    summaries = []
    query = "SELECT SUMMARY FROM METADATA WHERE ID = %s"

    for metadata_id in metadata_ids:
        cursor.execute(query, (metadata_id,))
        result = cursor.fetchone()
        if result:
            summaries.append(result[0])

    cursor.close()
    connection.close()
    return summaries

def get_authors_by_metadata_id(metadata_ids):
    connection = create_mysql_connection()
    cursor = connection.cursor()

    authors = []
    query = "SELECT AUTHOR FROM METADATA WHERE ID = %s"

    for metadata_id in metadata_ids:
        cursor.execute(query, (metadata_id,))
        result = cursor.fetchone()
        if result:
            authors.append(result[0])

    cursor.close()
    connection.close()
    return authors

def group_chunks(results):
    grouped_results = {}
    for author, file_name, summary, chunk in results:
        key = (author, file_name, summary)
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(chunk)

    final_results = [[author, file_name, summary, tuple(chunks)] for (author, file_name, summary), chunks in grouped_results.items()]
    return final_results

def submit():
    user_input = st.session_state.user_input
    st.session_state.messages.append({"role": "user", "content": user_input})

    context = "You are an advanced AI chatbot designed to assist users with their questions by providing relevant information extracted from a Research Paper database, supplied in the Additional Context section. The database contains metadata, summaries, and text chunks from a wide range of academic papers."
    objective = "Provide accurate, helpful, and contextually relevant responses to user queries by leveraging the information available in the Research Paper database. Aim to enhance the user's understanding of the queried topics. You may supplement this with your own knowledge, but primarily use the additional context in the research papers."
    structure = "Conversational and informative. Start with a direct response to the user's query, followed by additional context or information from the research papers to support the answer."
    tone = "Friendly, professional, and informative."
    audience = "Researchers, students, and the general public who are seeking detailed and reliable information from academic papers."
    result = "A clear, concise, and accurate response to the user's question, supported by relevant excerpts and metadata from the Research Paper database. You will specifically mention titles of Research Papers and their authors when appropriate."

    cleaned_query = remove_stopwords(user_input)
    st.session_state.cleaned_query = cleaned_query

    elastic_results = []
    for word in cleaned_query.split():
        elastic_results.extend(search_documents_in_elasticsearch(word))
    
    query_vector = vectorize_query(user_input)
    index, chunk_ids = get_faiss_index()
    faiss_results = search_chunks_in_faiss(query_vector, index, chunk_ids)

    normalized_faiss_scores = normalize_scores(faiss_results, 'faiss')
    normalized_elastic_scores = normalize_scores(elastic_results, 'elastic')
    combined_scores = combine_and_sort_scores(normalized_faiss_scores, normalized_elastic_scores)

    retrieved_chunk_ids, final_scores = extract_ids_and_scores(combined_scores)

    final_chunk_texts = get_chunk_text_by_id(retrieved_chunk_ids)
    final_doc_ids = get_doc_id_by_chunk_id(retrieved_chunk_ids)
    final_metadata_ids = get_metadata_primary_ids(final_doc_ids)
    file_names = get_file_name_by_doc_id(final_doc_ids)
    final_summaries = get_summaries_by_metadata_id(final_metadata_ids)
    final_authors = get_authors_by_metadata_id(final_metadata_ids)

    results = list(zip(final_authors, file_names, final_summaries, final_chunk_texts))

    grouped_results = group_chunks(results)
    grouped_results = grouped_results[:3]

    additional_context = "\n\n".join([f"Author: {result[0]}, Title: {result[1]}, Summary: {result[2]}, Excerpts: {', '.join(result[3][:3])}" for result in grouped_results])
    
    response = get_response(st.session_state.messages, context, objective, structure, tone, audience, result, additional_context)
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.session_state.user_input = ''

st.text_input("You: ", key="user_input", on_change=submit)

for message in st.session_state.messages:
    if message['role'] != 'system':
        role = "User" if message['role'] == 'user' else "Bot"
        st.write(f"{role}: {message['content']}")

if 'results' in st.session_state:
    st.write("Results:")
    st.write(st.session_state.results)
