from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from langchain.output_parsers import PydanticOutputParser

# TASK: Generate notes and quiz from the given text
# STRATEGY: Use parallel chain to generate notes and quiz separately and then merge them into a single document
# OUTPUT: A structured document containing notes and quiz questions with answers

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/QwQ-32B",
    task="text-generation",
)

model1 = ChatHuggingFace(
    llm=llm,
    temprature=0.8   
)

model2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8
)


class Quiz(BaseModel):
    question:str = Field(description="The question of the quiz")
    answer:str = Field(description="The answer of the question")

class Response(BaseModel):
    notes:str = Field(description="Generated notes based on the document")
    quiz: List[Quiz] = Field(description="Generated Quiz based on the document")

parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Response) 

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short questoins and answer from the following text /m {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document as the valid given format \n notes: {notes} and quiz: {quiz} \n {format_instructions}",
    input_variables=["notes" , "quiz"],
    partial_variables={"format_instructions" : pydantic_parser.get_format_instructions()}
)

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model1 | parser
})

merged_chain = prompt3 | model2 | pydantic_parser

chain = parallel_chain | merged_chain

text = """
Basically, Reach is just a fancy collection of Javascript. We all know that javascript is the catalyst that runs the internet and web pages. The larger and more complex your programs get, the harder it is to keep you javascript organized and scaleable. React addresses this issues with components. A React component is either a javascript class or function that renders JSX. (Remember this sentence. This is React in a nutshell. We will be breaking this sentence down throughout the article.) Components allow us to easily organize and scale our code. With each major section of your front-end application you can have a component. You can organize these components in folders. This way whenever you want to change a part of your application you know exactly where to look. One great thing about components is you can render a component as many times as you want. You can even have each render of the same component show greatly different things.(More on this later) React components is what allows you to make your application dynamic and be able to use the same code over and over.
Javascript is the engine that powers React, components is the transmission that makes React go. Let's go a little deeper into components. There are some main characteristics that help take React Components to the next level. We will talk about State, Props, JSX, and Lifecycle methods. Reminder, I am going to be as brief as I can. Look up more detail later on when you want to fully uncover these topics.
"""

result:Response = chain.invoke({"text" : text})

print(result.notes , end='\n\n\n')

for i,j in enumerate(result.quiz):
    print(f"Question {i} : ", j.question)

j = 0
for i,j in enumerate(result.quiz):
    print(f"Answers for the que {i} : " , j.answer)