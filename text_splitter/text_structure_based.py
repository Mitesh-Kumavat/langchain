from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
This is a long text that needs to be split into smaller chunks. The text contains multiple sentences and paragraphs. The goal is to create a text splitter that can handle different structures within the text, such as paragraphs, sentences, and even bullet points.
The text splitter should be able to identify these structures and split the text accordingly. For example, if the text contains bullet points, the splitter should create separate chunks for each bullet point. Similarly, if the text contains paragraphs, the splitter should create separate chunks for each paragraph.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:\n{chunk}\n")
