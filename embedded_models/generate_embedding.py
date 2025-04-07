from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

print("Model loaded successfully.")

docs  = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
]

text = "New Delhi is the capital of India."

res = embedding.embed_documents(docs)
print(str(res))

res = embedding.embed_query(text)
print(str(res))