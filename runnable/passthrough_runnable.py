from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel

# RunnablePasssthrough is a simple passthrough runnable that takes an input and returns it as output.
# It is useful for debugging and testing purposes.
# In the previous example of RunnableSequence, we created a joke and pass it to the next step to explain it. So we are not be able to see the joke in the output. Only the explanation is shown.
# In this example we will use RunnablePassthrough to see the joke, and explaination of the joke in the output.

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="explain this joke {joke}",
    input_variables=["joke"],
)

parser = StrOutputParser()

joke_generator = RunnableSequence(prompt, model, parser )
joke_explainer = RunnableSequence(prompt2, model, parser )

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': joke_explainer
})

final_chain = RunnableSequence(
    joke_generator,
    parallel_chain
)

result = final_chain.invoke({"topic" : "cats"})

print(result)
