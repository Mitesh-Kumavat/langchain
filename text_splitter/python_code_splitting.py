from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text="""
class MyClass:
    def __init__(self, name):
        self.name = name
        self.value = 0
        
    def increment(self, amount):
        self.value += amount
        
    def get_value(self):
        return self.value
        
#Example Usage
my_instance = MyClass("Example")
my_instance.increment(5)
if my_instance.get_value() > 0:
    print(f"{my_instance.name} has a positive value: {my_instance.get_value()}")
else:
    print(f"{my_instance.name} has a non-positive value: {my_instance.get_value()}")
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size=240,
    chunk_overlap=0,
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")


