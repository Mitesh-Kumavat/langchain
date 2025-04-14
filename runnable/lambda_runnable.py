from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence,RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# RunnableLambda can convert any python function into a runnablefrom dotenv import load_dotenv

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9
)

def word_counter(text):
    """Counts the number of words in a text."""
    return len(text.split())

prompt = PromptTemplate(
    template="generate a joke about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

joke_generator = RunnableSequence(prompt, model, parser)

parallel_chain =RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count":  RunnableLambda(word_counter),
})

# The same code can work with lambda functions as well
# 
# parallel_chain =RunnableParallel({
#     "joke": RunnablePassthrough(),
#     "word_count":  RunnableLambda(lambda text: len(text.split())),
# })

final_chain = RunnableSequence(joke_generator, parallel_chain)

result = final_chain.invoke({"topic": "AI"})

print(result)