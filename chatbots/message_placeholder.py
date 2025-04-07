from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1,
)

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}"),
])

chat_history =[]

with open("chat_history.txt", "r") as file:
    chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    "query": "What is the status of my order ?"
})
result = model.invoke(prompt)
print(result.content)

