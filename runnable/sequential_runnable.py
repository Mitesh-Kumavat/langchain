from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Explain the joke {joke}",
    input_variables=["joke"],
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser , prompt2, model, parser)

result = chain.invoke({"topic" : "cats"})

print(result)