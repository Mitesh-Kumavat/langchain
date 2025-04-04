from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
)

chat_history =[
    SystemMessage(content="You are a helpful assistant."),
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting chat...")
        break

    chat_history.append(HumanMessage(content=user_input))

    response = model(messages=chat_history)

    chat_history.append(AIMessage(content=response.content))

    print(f"AI: {response.content}")
