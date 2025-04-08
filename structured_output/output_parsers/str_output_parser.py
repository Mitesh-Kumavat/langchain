from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template='write a 5 line summary on the following text. \n {text}',
    input_variables=["text"],
)

model = ChatHuggingFace(llm=llm)

# NOTE: if you do not use Chain and str output parser, you can do it like this:

# prompt1 = template1.invoke({"topic": "Gaming Mouse"})
# res = model.invoke(prompt1)
# prompt2 = template2.invoke({"text": res.content})
# ans  = model.invoke(prompt2)
# print(ans.content)


# with the help of Chain and str output parser, you can do it like this:
parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Black hole"})
print(result)