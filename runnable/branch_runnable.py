from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough, RunnableBranch
from dotenv import load_dotenv

load_dotenv()

# RunnableBranch can be used to run a particular runnable when certain conditions are met.
# For example, if you have a chain that generates a report and another chain that summarizes the report,
# we will first generate the report and if it is longer than 200 words then only we will summarize it.
# In this case, we will use RunnableBranch to run the summarization chain only if the report is longer than 200 words.

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.9
)

prompt = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="summarize the following: {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()


report_generator = RunnableSequence(prompt, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200 , RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough() # This will be run if the condition is not met, and it will just pass the input to the next runnable.
)

final_chain = RunnableSequence(report_generator, branch_chain)

result = final_chain.invoke({"topic": "AI in healthcare"})
print(result)