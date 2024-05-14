import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import mysql.connector
from mysql.connector import Error
from keys import mysql_password
from elasticsearch import Elasticsearch
import pandas as pd

# Constants
HOST_NAME = 'localhost'
USER_NAME = 'root'
DATABASE_NAME = 'ARXIVPaperDB'
USER_PASSWORD = mysql_password

# Models
model = SentenceTransformer('all-MiniLM-L6-v2')
es = Elasticsearch("http://localhost:9200")

#Functions

class MySQLDatabase:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def create_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return connection
        except Error as e:
            print(f"The error '{e}' occurred")
            return None

    def get_chunk_text_by_id(self, chunk_ids):
        connection = self.create_connection()
        cursor = connection.cursor()

        format_strings = ','.join(['%s'] * len(chunk_ids))
        query = f"SELECT TEXT FROM CHUNK WHERE ID IN ({format_strings})"
        cursor.execute(query, tuple(chunk_ids))
        results = cursor.fetchall()
        texts = [result[0] for result in results]

        cursor.close()
        connection.close()

        return texts

    def get_doc_id_by_chunk_id(self, chunk_ids):
        connection = self.create_connection()
        cursor = connection.cursor()

        format_strings = ','.join(['%s'] * len(chunk_ids))
        query = f"SELECT DOCUMENT_ID FROM CHUNK WHERE ID IN ({format_strings})"
        cursor.execute(query, tuple(chunk_ids))
        results = cursor.fetchall()
        doc_ids = [result[0] for result in results]

        cursor.close()
        connection.close()

        return doc_ids

    def get_doc_id_by_metadata_id(self, metadata_ids):
        if not metadata_ids:
            return []

        connection = self.create_connection()
        cursor = connection.cursor()

        format_strings = ','.join(['%s'] * len(metadata_ids))
        query = f"SELECT DOCUMENT_ID FROM METADATA WHERE ID IN ({format_strings})"
        cursor.execute(query, tuple(metadata_ids))
        results = cursor.fetchall()
        doc_ids = [result[0] for result in results]

        cursor.close()
        connection.close()

        return doc_ids

    def get_metadata_primary_ids(self, document_ids):
        connection = self.create_connection()
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

    def get_file_name_by_doc_id(self, document_ids):
        connection = self.create_connection()
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

    def get_file_extensions_by_doc_id(self, document_ids):
        connection = self.create_connection()
        cursor = connection.cursor()

        file_extensions = []
        query = "SELECT DOC_TYPE FROM DOCUMENT WHERE ID = %s"

        for document_id in document_ids:
            cursor.execute(query, (document_id,))
            result = cursor.fetchone()
            if result:
                file_extensions.append(result[0])

        cursor.close()
        connection.close()
        return file_extensions

    def get_content_lengths_by_metadata_id(self, metadata_ids):
        connection = self.create_connection()
        cursor = connection.cursor()

        content_lengths = []
        query = "SELECT CONTENT_LENGTH FROM METADATA WHERE ID = %s"

        for metadata_id in metadata_ids:
            cursor.execute(query, (metadata_id,))
            result = cursor.fetchone()
            if result:
                content_lengths.append(result[0])

        cursor.close()
        connection.close()
        return content_lengths

    def get_summaries_by_metadata_id(self, metadata_ids):
        connection = self.create_connection()
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

    def get_creation_dates_by_metadata_id(self, metadata_ids):
        connection = self.create_connection()
        cursor = connection.cursor()

        creation_dates = []
        query = "SELECT CREATION_DATE FROM METADATA WHERE ID = %s"

        for metadata_id in metadata_ids:
            cursor.execute(query, (metadata_id,))
            result = cursor.fetchone()
            if result:
                creation_dates.append(result[0])

        cursor.close()
        connection.close()
        return creation_dates

    def get_authors_by_metadata_id(self, metadata_ids):
        connection = self.create_connection()
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

class ElasticsearchSearch:
    def __init__(self, es_instance):
        self.es = es_instance

    def search_documents_and_get_top_results(self, index_name, query_text, top_k=10):
        try:
            response = self.es.search(
                index=index_name,
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

    def search_authors_and_get_top_results(self, index_name, query_text, top_k=10):
        try:
            response = self.es.search(
                index=index_name,
                body={
                    "query": {
                        "match": {
                            "author": query_text
                        }
                    }
                },
                size=top_k
            )

            top_results = [(hit['_source']['metadata_id'], hit['_score']) for hit in response['hits']['hits']]
            return top_results
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

class FAISSIndex:
    def __init__(self, model):
        self.model = model

    def vectorize_query(self, query):
        query_vector = self.model.encode(query, convert_to_tensor=False)
        return query_vector

    def get_embeddings(self, connection):
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

    def search_chunks(self, query_vector, index, document_ids, top_k=10):
        query_vector = np.array([query_vector])

        distances, indices = index.search(query_vector, top_k)
        top_chunks = [(document_ids[idx], dist) for idx, dist in zip(indices[0], distances[0])]
        
        return top_chunks

class SearchEngine:
    def __init__(self, db, es_search, faiss_index):
        self.db = db
        self.es_search = es_search
        self.faiss_index = faiss_index

    def normalize_scores(self, results, search_type):
        if not results:
            return []

        scores = [result[1] for result in results]
        
        if search_type == 'faiss':
            max_score = max(scores)
            min_score = min(scores)
            normalized_scores = [(results[i][0], 1 - (scores[i] - min_score) / (max_score - min_score)) for i in range(len(results))]
        elif search_type == 'elastic':
            max_score = max(scores)
            min_score = min(scores)
            normalized_scores = [(results[i][0], (scores[i] - min_score) / (max_score - min_score)) for i in range(len(results))]
        else:
            raise ValueError("Invalid score_type. Use 'faiss' or 'elastic'.")

        return normalized_scores

    def extract_ids_and_scores(self, sorted_scores):
        retrieved_ids = [item[0] for item in sorted_scores]
        final_scores = [item[1] for item in sorted_scores]
        return retrieved_ids, final_scores

class StreamlitApp:
    def __init__(self, search_engine):
        self.search_engine = search_engine

    def main(self):
        st.title("ArXiv Paper Search Engine")
        query = st.text_input("Enter a query:")
        if query:
            st.write(f"Search for: {query}")

        col1, col2 = st.columns(2)

        with col1:
            search_documents_button = st.button("Search Documents")
        with col2:
            search_authors_button = st.button("Search Authors")

        if 'sorted_df' not in st.session_state:
            st.session_state.sorted_df = pd.DataFrame()

        if 'author_df' not in st.session_state:
            st.session_state.author_df = pd.DataFrame()

        # Create placeholders for displaying results
        document_placeholder = st.empty()
        author_placeholder = st.empty()

        # Document Search
        if search_documents_button:
            st.session_state.sorted_df = pd.DataFrame()  # Reset document search results
            author_placeholder.empty()  # Clear previous author search results

            top_results = self.search_engine.es_search.search_documents_and_get_top_results('arxiv_paper_chunks', query)

            connection = self.search_engine.db.create_connection()
            query_vector = self.search_engine.faiss_index.vectorize_query(query)
            index, chunk_ids = self.search_engine.faiss_index.get_embeddings(connection)
            top_chunks = self.search_engine.faiss_index.search_chunks(query_vector, index, chunk_ids)

            normalized_faiss_scores = self.search_engine.normalize_scores(top_chunks, 'faiss')
            normalized_elastic_scores = self.search_engine.normalize_scores(top_results, 'elastic')

            combined_scores = {}
            for chunk_id, score in normalized_faiss_scores:
                combined_scores[chunk_id] = score
            for chunk_id, score in normalized_elastic_scores:
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] += score
                else:
                    combined_scores[chunk_id] = score

            sorted_combined_scores = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            retrieved_chunk_ids, final_scores = self.search_engine.extract_ids_and_scores(sorted_combined_scores)

            final_chunk_texts = self.search_engine.db.get_chunk_text_by_id(retrieved_chunk_ids)
            final_doc_ids = self.search_engine.db.get_doc_id_by_chunk_id(retrieved_chunk_ids)
            final_metadata_ids = self.search_engine.db.get_metadata_primary_ids(final_doc_ids)
            file_names = self.search_engine.db.get_file_name_by_doc_id(final_doc_ids)
            file_extensions = self.search_engine.db.get_file_extensions_by_doc_id(final_doc_ids)
            final_content_lengths = self.search_engine.db.get_content_lengths_by_metadata_id(final_metadata_ids)
            final_summaries = self.search_engine.db.get_summaries_by_metadata_id(final_metadata_ids)
            final_creation_dates = self.search_engine.db.get_creation_dates_by_metadata_id(final_metadata_ids)
            final_authors = self.search_engine.db.get_authors_by_metadata_id(final_metadata_ids)

            data = {
                'Chunk ID': retrieved_chunk_ids,
                'Score': final_scores,
                'Chunk Text': final_chunk_texts,
                'Document ID': final_doc_ids,
                'Metadata ID': final_metadata_ids,
                'File Name': file_names,
                'File Extension': file_extensions,
                'Content Length': final_content_lengths,
                'Summary': final_summaries,
                'Creation Date': final_creation_dates,
                'Author': final_authors
            }

            document_df = pd.DataFrame(data)

            grouped_df = document_df.groupby('Document ID').agg({
                'Chunk ID': 'first',
                'Score': 'sum',
                'Chunk Text': 'first',
                'Metadata ID': 'first',
                'File Name': 'first',
                'File Extension': 'first',
                'Content Length': 'first',
                'Summary': 'first',
                'Creation Date': 'first',
                'Author': 'first'
            }).reset_index()

            st.session_state.sorted_df = grouped_df.sort_values(by='Score', ascending=False)

            with document_placeholder.container():
                st.markdown("---")
                if not st.session_state.sorted_df.empty:
                    for index, row in st.session_state.sorted_df.iterrows():
                        st.markdown(
                            f"""
                            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
                                <div style="font-size: small; color: #FFFFFF;">Score: {row['Score']} | Date created: {row['Creation Date']}</div>
                                <br>
                                <div style="font-size: large; font-weight: bold; margin-top: 5px;">{row['File Name']}</div>
                                <br>
                                <div style="font-size: small; color: #555; margin-top: 5px;"><span style="background-color: #ffff00;">{row['Chunk Text']}</span></div>
                                <br>
                                <div style="font-size: small; color: #FFFFFF; margin-top: 5px;">Summary: {row['Summary']}</div>
                                <br>
                                <div style="font-size: small; color: #FFFFFF;">Author: {row['Author']}</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.markdown("<hr>", unsafe_allow_html=True)

                    st.write(st.session_state.sorted_df)

        # Author Search
        if search_authors_button:
            st.session_state.author_df = pd.DataFrame()  # Reset author search results
            document_placeholder.empty()  # Clear previous document search results

            top_results = self.search_engine.es_search.search_authors_and_get_top_results('arxiv_authors', query)
            retrieved_metadata_ids, final_scores = self.search_engine.extract_ids_and_scores(top_results)
            
            final_doc_ids = self.search_engine.db.get_doc_id_by_metadata_id(retrieved_metadata_ids)
            file_names = self.search_engine.db.get_file_name_by_doc_id(final_doc_ids)
            file_extensions = self.search_engine.db.get_file_extensions_by_doc_id(final_doc_ids)
            final_content_lengths = self.search_engine.db.get_content_lengths_by_metadata_id(retrieved_metadata_ids)
            final_summaries = self.search_engine.db.get_summaries_by_metadata_id(retrieved_metadata_ids)
            final_creation_dates = self.search_engine.db.get_creation_dates_by_metadata_id(retrieved_metadata_ids)
            final_authors = self.search_engine.db.get_authors_by_metadata_id(retrieved_metadata_ids)

            data = {
                'Metadata ID': retrieved_metadata_ids,
                'Score': final_scores,
                'Document ID': final_doc_ids,
                'File Name': file_names,
                'File Extension': file_extensions,
                'Content Length': final_content_lengths,
                'Summary': final_summaries,
                'Creation Date': final_creation_dates,
                'Author': final_authors
            }

            author_df = pd.DataFrame(data)
            st.session_state.author_df = author_df.sort_values(by='Score', ascending=False)

            with author_placeholder.container():
                st.markdown("---")
                if not st.session_state.author_df.empty:
                    for index, row in st.session_state.author_df.iterrows():
                        st.markdown(
                            f"""
                            <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
                                <div style="font-size: small; color: #FFFFFF;">Score: {row['Score']} | Date created: {row['Creation Date']}</div>
                                <br>
                                <div style="font-size: large; font-weight: bold; margin-top: 5px;">{row['File Name']}</div>
                                <br>
                                <div style="font-size: small; color: #FFFFFF; margin-top: 5px;">Summary: {row['Summary']}</div>
                                <br>
                                <div style="font-size: small; color: #555;"><span style="background-color: #ffff00;">Author: {row['Author']}</span></div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        st.markdown("<hr>", unsafe_allow_html=True)

                    st.write(st.session_state.author_df)

#MAIN

if __name__ == "__main__":
    db = MySQLDatabase(HOST_NAME, USER_NAME, USER_PASSWORD, DATABASE_NAME)
    es_search = ElasticsearchSearch(es)
    faiss_index = FAISSIndex(model)
    search_engine = SearchEngine(db, es_search, faiss_index)
    app = StreamlitApp(search_engine)
    app.main()