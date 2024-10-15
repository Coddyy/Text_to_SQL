import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# GEMINI PRO
import google.generativeai as genai
genai.configure(api_key=GOOGLE_API_KEY)
#llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85)
#llm = genai.GenerativeModel("gemini-pro")

# POSTGRES
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:postgres@localhost/dvdrental")
#print(db.get_table_info())

def get_schema(_):
    return db.get_table_info()


template = """
based on the schema below, write a SQL query that would answer the user's question!
{schema}

Question: {question}
SQL Query
"""

prompt = ChatPromptTemplate.from_template(template)

prompt.format(schema="my schema", question="how many tables are there?")


sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm #.bind(stop="\nSQL Result:")
    | StrOutputParser()
)

result = sql_chain.invoke({"question": "How many customers are there?"})


def run_query(query):
    return db.run(query)


user_template = """
Based on the schema below, question, sql query and sql response write a natural language response!
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}
"""

user_prompt = ChatPromptTemplate.from_template(user_template)

full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda variables: run_query(variables['query'].replace('```sql', '').strip('`'))
    )
    | user_prompt
    | llm
    | StrOutputParser()
)

def get_gemini_response(question):
    return full_chain.invoke({"question": question})
    #print(result)


st.set_page_config(page_title="LLM NLP To SQL")
st.header("Chat With DB - Gemini Pro")
question =  st.text_input("Input", key="input")
submit = st.button("Ask a question")

if submit:
    response = get_gemini_response(question)
    st.header(response)
