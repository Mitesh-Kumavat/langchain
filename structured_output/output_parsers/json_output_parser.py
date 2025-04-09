from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import JsonOutputParser 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

model = ChatHuggingFace(
    llm=llm,
    temperature=0.7,
)

parser = JsonOutputParser()

template1 = PromptTemplate(
    template='Give me the name, age and city of fictional person \n  {format_instructions}',
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

chain = template1 | model| parser
result = chain.invoke({})

print(result)