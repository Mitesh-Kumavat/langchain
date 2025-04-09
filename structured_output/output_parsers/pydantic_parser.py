from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

model = ChatHuggingFace(
    llm=llm,
    temperature=0.7,
)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person", gt=0)
    city: str = Field(description="City of the person")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me the name, age and city of fictional {place} person \n  {format_instructions}',
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = template | model | parser

result = chain.invoke({"place": "India"})
print(result)
json_res = result.json()
print(json_res)