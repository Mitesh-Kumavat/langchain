from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('node-js.pdf')

docs = loader.load()

print(len(docs))
print(docs)

print('--- First Page ---')
print(docs[0].page_content)


# Limitations of PyPDFLoader:
'''
1. It may not handle all PDF formats: PyPDFLoader may not be able to extract text from all types of PDF files, especially those with complex layouts or embedded fonts.
2. It may not preserve formatting: The extracted text may not retain the original formatting of the PDF, such as font styles, colors, and images.
3. It may not handle large files efficiently: PyPDFLoader may struggle with very large PDF files, leading to performance issues or memory errors.
'''

# PDF loaders and there use cases:
'''
1. PyPDFLoader : simple, clean PDF
2. PDFPlumberLoader: Pdf with tables and columns
3. UnstructuredPDFLoader or AmazonTextractPDFLoader: Unstructured PDF with images and scanned PDF
4. PyMupdfLoader: PDF with layout and image data
5. UnstructuredPDFLoader: Best structured extraction
'''

