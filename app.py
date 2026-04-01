from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
 
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SequentialChain 
load_dotenv()
 
app = Flask(__name__)
 
llm = ChatOpenAI(model_name="gpt-4o-mini")
 
# Chains
summary_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["tasks"],
        template="Summarize the following tasks: {tasks}"
    ),
    output_key="summary"
)
 
classification_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["summary"],
        template="Classify tasks into Work or Personal: {summary}"
    ),
    output_key="category"
)
 
priority_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["tasks"],
        template="""
Assign priority (High, Medium, Low):
 
{tasks}
"""
    ),
    output_key="priority"
)
 
chain = SequentialChain(
    chains=[summary_chain, classification_chain, priority_chain],
    input_variables=["tasks"],
    output_variables=["summary", "category", "priority"]
)
 
# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        tasks = request.form["tasks"]
        result = chain({"tasks": tasks})
 
    return render_template("index.html", result=result)
 
if __name__ == "__main__":
    app.run(debug=True)