from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# TASK: Generate a detailed report on the given topic and summarize it in 5 points
# STRATEGY: Use sequential chain to generate a report and then summarize it

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

model = ChatHuggingFace(
    llm=llm,
    temperature=0.7,
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate detailed report on the {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Summarize the {report} in 5 points",
    input_variables=["report"],
)


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic" : "Black hole"})

print(result)

# How chain looks like 
chain.get_graph().print_ascii()