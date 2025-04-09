from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

model = ChatHuggingFace(
    llm=llm,
    temperature=0.7,
)

schema = [
    ResponseSchema(name="fact_1", description="A fact about the topic"),
    ResponseSchema(name="fact_2", description="A fact about the topic"),
    ResponseSchema(name="fact_3", description="A fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 fact about {topic} \n  {format_instructions}',
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

topic = "Black hole"

chain = template | model | parser
result = chain.invoke({"topic": topic})
print(result, end="\n\n")
print("Fact1:", result["fact_1"])