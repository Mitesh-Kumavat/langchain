from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9
)

prompt = PromptTemplate(
    template="generate a 2 line tweet about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a 2 line linkedin about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt, model, parser),
    "linkedin": RunnableSequence(prompt2, model, parser),
})

result = parallel_chain.invoke({"topic": "AI"})

print(result)
print("------Print separate results------")
print(result['tweet'])
print("----------------------------------")
print(result['linkedin'])