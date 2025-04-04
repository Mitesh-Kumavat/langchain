from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful {domain} expert."),
    ("human", "Explain in simple term what is {topic}."),
])

prompt = chat_template.invoke({
    "domain": "Cricket",
    "topic": "Bowling"
})

print(prompt)