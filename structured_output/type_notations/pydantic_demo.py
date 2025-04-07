from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Use the ChatOpenAI model for structured output 
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro"
)

class User(BaseModel):
    username: str = Field(..., title="Username", description="The user's unique username")
    email: EmailStr = Field(..., title="Email Address", description="The user's email address")
    age: int = Field(..., gt=0, le=120, title="Age", description="The user's age in years")
    bio: str = Field(None, title="Biography", description="A short biography of the user")
    

model.with_structured_output(User)

user_data = """
    I am a software engineer with over 10 years of experience in web development. I love coding and enjoy learning new technologies. My hobbies include hiking, reading, and playing video games. I am also an avid traveler and have visited over 20 countries. My name is john doe and i am currently 36 years old, my email is johndoe@gmail.com and my username is johndoe123.I am currently working at a tech company as a senior software engineer. I have a passion for creating innovative solutions and improving user experiences. I am always looking for new challenges and opportunities to grow in my career.0 
"""

result = model.invoke(user_data)

print(result.content)