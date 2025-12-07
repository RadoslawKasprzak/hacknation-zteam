import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector


def preprocess_and_chunk_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Loads a document and splits it into semantically coherent chunks, preserving metadata.

    :param file_path: Path to the document (e.g., PDF).
    :param chunk_size: Maximum chunk size in characters.
    :param chunk_overlap: The degree of overlap between chunks.
    :return: List of Document objects (chunks).
    """
    print(f"Loading document: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Hierarchy of separators
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["document_id"] = os.path.basename(file_path)

    print(f"Preprocessing complete. Created {len(chunks)} chunks.")
    return chunks


def save_chunks_to_gcp_vector_db(chunks: List[Document], connection_string: str, collection_name: str) -> Optional[
    PGVector]:
    """
    Generates embeddings using Vertex AI Embeddings and saves them to the PGVector database.

    :param chunks: List of processed Document objects.
    :param connection_string: PostgreSQL connection URI.
    :param collection_name: Name of the collection/table in PGVector.
    :return: An instance of PGVector if successful, otherwise None.
    """
    if not chunks:
        print("No chunks to save.")
        return None

    print("Initializing Vertex AI Embeddings model...")
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        task_type="RETRIEVAL_DOCUMENT"
    )

    print(f"Starting write to PGVector for {len(chunks)} chunks...")

    try:
        vector_db = PGVector.from_documents(
            embedding=embeddings_model,
            documents=chunks,
            connection_string=connection_string,
            collection_name=collection_name,
            use_jsonb=True
        )

        print(f"Successfully wrote {len(chunks)} vectors to PGVector (Collection: {collection_name}).")
        return vector_db

    except Exception as e:
        print(f"Critical error writing to PGVector. Error: {e}")
        return None
