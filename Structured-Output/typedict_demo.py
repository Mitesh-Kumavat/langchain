from typing import TypedDict, Annotated,Literal, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# NOTE: This code will not work because the with_structured_output method is not available in the current version of langchain_google_genai instead you can try this example with ChatOpenAI.
# This is a hypothetical example to demonstrate how you might use TypedDict with structured output in a future version of the library.


# Use the ChatOpenAI model for structured output 


model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro"
)

# Define a TypedDict for the structured output
class ReviewAnalysis(TypedDict):
    sentiment: Annotated[Literal["pos","neg"], "sentiment of the review"]
    summary: Annotated[str, "a brief summary of the review"]
    pros: Annotated[Optional[list[str]], "list of positive aspects"]
    cons: Annotated[Optional[list[str]], "list of negative aspects"]


model.with_structured_output(ReviewAnalysis)

review = """
The iPhone 14 Pro Max is a powerful and feature-rich smartphone that offers an exceptional user experience. With its stunning Super Retina XDR display, A16 Bionic chip, and impressive camera system, it stands out as one of the best devices on the market. The battery life is also commendable, lasting all day with regular use. However, the high price tag may be a drawback for some users. Overall, the iPhone 14 Pro Max is a top-tier smartphone that delivers on performance and design.
"""

result = model.invoke(review)

print(result.content)