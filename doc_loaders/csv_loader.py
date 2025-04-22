from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("sample.csv")

documents = loader.load()

print(f"Loaded {len(documents)} documents from CSV file.")
# Display the loaded documents
print(documents[0])