from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# TextLoader is a document loader that loads text files.

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.5,
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='write a summary of the following : {text}',
    input_variables=['text'],
)

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()
# now docs has the list of documents loaded from the text file
# Print the documents to see the content
# print(docs)
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompt | model | parser
result = chain.invoke({"text":docs[0].page_content})
print(result)
