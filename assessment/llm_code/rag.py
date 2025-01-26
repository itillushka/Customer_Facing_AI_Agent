from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import pandas as pd
import os

class DentalServiceRAG:
    def __init__(self, data_folder="../data", filename="dental_clinic_data.csv", embedding_model="text-embedding-ada-002"):
        """
        Initialize the RAG system for the dental service dataset.
        :param data_folder: Relative folder where the dataset is located.
        :param filename: Name of the dataset file.
        :param embedding_model: The embedding model to use.
        """
        self.data_folder = data_folder
        self.filename = filename
        self.dataset_path = os.path.abspath(os.path.join(data_folder, filename))
        self.embedding_model = embedding_model
        self.vector_store = None

    def load_and_index_data(self):
        """Load data from CSV, convert to documents, and index with FAISS."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")
        
        data = pd.read_csv(self.dataset_path)

        # Convert to LangChain Document format
        documents = []
        for _, row in data.iterrows():
            content = f"Service: {row['Service Name']}\nDescription: {row['Description']}\nPrice: {row['Price']}\nSpecialist: {row['Specialist']}\nPreparation: {row['Preparation']}\nDuration: {row['Duration (mins)']} minutes."
            documents.append(Document(page_content=content))

        # Create embeddings and index using FAISS
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.vector_store = FAISS.from_documents(documents, embeddings)

    def retrieve(self, query, top_k=3):
        """
        Retrieve the most relevant documents for a given query.
        :param query: The query to search the dataset for.
        :param top_k: Number of top documents to return.
        :return: List of top-k relevant document content.
        """
        if not self.vector_store:
            raise ValueError("The data has not been indexed. Call load_and_index_data first.")

        # Ensure top_k is an integer
        top_k = int(top_k)

        results = self.vector_store.similarity_search(query, k=top_k)
        return [result.page_content for result in results]
