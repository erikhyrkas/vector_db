import sys
import os
import numpy as np
import pickle
import gzip
from sentence_transformers import SentenceTransformer


class LocalVectorDb:
    def __init__(self, documents=None, embedding_model="sentence-transformers/gtr-t5-large"):
        self.sentence_transformer = SentenceTransformer(embedding_model)
        self.embeddings = None
        self.documents = documents
        if self.documents:
            embeddings = self.embed(self.documents)
            self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    def embed(self, docs):
        return np.array(self.sentence_transformer.encode(docs)).astype(np.float32)

    def cosine_similarity(self, query_vector):
        normalized_query_vector = query_vector / np.linalg.norm(query_vector)
        return np.dot(self.embeddings, normalized_query_vector.T)

    def save(self, db_file):
        data = {
            "embeddings": self.embeddings,
            "documents": self.documents
        }
        with gzip.open(db_file, "wb") as f:
            pickle.dump(data, f)

    def load(self, db_file):
        with gzip.open(db_file, "rb") as f:
            data = pickle.load(f)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.documents = data["documents"]

    def best_index(self, query_vector):
        return np.argsort(self.cosine_similarity(query_vector), axis=0)[-1:][::-1][0]

    def query(self, query):
        query_vector = self.embed([query])[0]
        return self.documents[self.best_index(query_vector)]


def read_text_files(directory):
    print("Reading text documents", end='')
    file_contents = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".adoc") or file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    print(".", end='')
                    file_contents.append(f.read())

    print("")
    print("Done.")
    return file_contents


def load_or_build_vector_db(documents_folder, db_file_name="documents.bin"):
    if os.path.exists(db_file_name):
        print("Loading existing vector database...")
        db = LocalVectorDb()
        db.load(db_file_name)
        print("Loaded.")
        return db

    print("Building vector database for first run.")
    all_contents = read_text_files(documents_folder)
    db = LocalVectorDb(all_contents)
    db.save(db_file_name)
    print("Created.")
    return db


def entry():
    documents_folder = './docs'
    if len(sys.argv) > 1:
        documents_folder = sys.argv[1]

    vector_db = load_or_build_vector_db(documents_folder)

    while True:
        user_prompt = input("> ")
        if user_prompt == "exit":
            break
        best_doc = vector_db.query(user_prompt)
        print(best_doc)


if __name__ == '__main__':
    entry()
