from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# docs = loader.load()
# print(docs[0].page_content)
# this will load all the documents in the directory and return a list of documents 

# we can use lazy_load to load the documents lazily it returns a generator
# this is useful when we have a large number of documents and we want to load them one by one

docs = loader.lazy_load()

for document in docs:
    print(document.page_content)