from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback.")


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)


parser = StrOutputParser()
pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback as positive or negative: \n {feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser

prompt_positive = PromptTemplate(
    template="Write an appropriate response to the following positive feedback: \n {feedback} ",
    input_variables=["feedback"],
)

prompt_negative = PromptTemplate(
    template="Write an appropriate response to the following negative feedback: \n {feedback} ",
    input_variables=["feedback"],
)

# conditional chain
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | parser),
    (lambda x: x.sentiment == "negative", prompt_negative | model | parser),
    RunnableLambda(lambda x: "could not find the sentiment")
)

feedback = "The best product ever! I love it. But the delivery was late."

chain = classifier_chain | branch_chain

try:    
    result = chain.invoke({"feedback": feedback})
    print(result)
except Exception as e:
    print(f"Error: {e}")