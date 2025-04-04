from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=1.1,
)

response = model.invoke("what are the models of Google Gemini like gemini-1.5-pro , or gemini-2.0-flash and many more... and what are their differences between them?")
print(response.content)