from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text ="""
Some of the most common text splitting methods include:
- Character-based splitting: This method splits text into chunks based on a specified number of characters. It is simple and effective for short texts, but may not preserve the meaning of the text.
- Word-based splitting: This method splits text into chunks based on a specified number of words. It is more effective than character-based splitting for longer texts, as it preserves the meaning of the text better.
- Sentence-based splitting: This method splits text into chunks based on sentence boundaries. It is effective for longer texts, as it preserves the meaning of the text and allows for better readability.
- Paragraph-based splitting: This method splits text into chunks based on paragraph boundaries. It is effective for longer texts, as it preserves the meaning of the text and allows for better readability.
- Token-based splitting: This method splits text into chunks based on a specified number of tokens. It is effective for longer texts, as it preserves the meaning of the text and allows for better readability.
- Custom splitting: This method allows for custom splitting based on specific criteria, such as keywords or regular expressions. It is effective for longer texts, as it preserves the meaning of the text and allows for better readability.
- Semantic splitting: This method splits text into chunks based on semantic meaning, such as topic or theme. It is effective for longer texts, as it preserves the meaning of the text and allows for better readability.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator="",
)

result = splitter.split_text(text)

print(result)

# using PyPDFLoader to load a PDF file and split it into chunks

loader = PyPDFLoader("node-js.pdf")

docs = loader.load()

result_with_loader = splitter.split_documents(docs)

print(result_with_loader)
print(f"Number of chunks: {len(result_with_loader)}")

