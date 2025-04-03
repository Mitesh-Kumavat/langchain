from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is a great batsman.He his known for his great batsmanship and aggressive nature.",
    "MS Dhoni is a great captain. He is known for his calmness and great captaincy.",
    "Rohit Sharma is a great opener. He is known for his great batting and aggressive nature.",
    "Sachin Tendulkar is a great batsman. He is known for his great batting and also known as 'God Of Cricket'.",    
    "Jasprit Bumrah is a great bowler. He is known for his great bowling and yorker balls.",
]

query = "Tell me about great bowler"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print(f"Query: {query}")
print(f"Answer: {documents[index]}")
print(f"Score: {score}")