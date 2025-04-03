from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
)

template = PromptTemplate(
    template="""
    You are a research paper summarizer.
    please summarize the given research paper: "{paper_input}" in "{style_input}" style and "{length_input}" length.
    if you dont know the answer, do not hulicinate, just decline that you dont know the answer and say "I am not sure about that".
    in the end, please provide a list of references for the research paper.
    Do not include any other information or explanation.
    """,
    input_variables=["paper_input", "style_input", "length_input"],
)



# Streamlit code ...
st.header("Research Chatbot")

paper_input= st.selectbox(
    "Select a Reseaarch paper",
    [ "Attention is all you need", "Random Paper", "Anatomy of Human Heart" , "Prostate Cancer"],
)

style_input = st.selectbox(
    "Select Explanation style",
    [ "Begineer friendly", "Mathematical","Code" ,"Technical", "Simplified"],
)

length_input = st.selectbox(
    "Select Explanation length",
    [ "Short(1-2 para)", "Medium(3-5 para)", "Long(5+ para)"],
)


prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input,
})


if st.button("Summarize"):
    result = model.invoke(prompt)
    st.write(result.content)
    
