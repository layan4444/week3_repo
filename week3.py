from dotenv import load_dotenv
import os
from langchain_core.tools import Tool
from langchain.agents import create_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SequentialChain


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

def count_words(text: str) -> str:
   return f"Number of words: {len(text.split())}"
 
def summarize(text: str) -> str:
   return "Summary: The day included studying and completing assingment."
 
tools = [
Tool(name="word_counter", func=count_words, description="Counts words in a string"),
Tool(name="summarizer", func=summarize, description="Summarizes daily tasks")
]
 
llm = ChatOpenAI(model="gpt-4o-mini", api_key= api_key )
agent = create_agent(llm, tools=tools)
 
user_input = input("Enter your task:\n")
 
 
result = agent.invoke({
"messages": [{"role": "user", "content": user_input}]
})
 
print("\nAgent Result:")
print(result["messages"][-1].content)

prompt = PromptTemplate(
    input_variables=["tasks"],
    template="Summarize the following tasks: {tasks}"
)
 
llm = ChatOpenAI(model_name="gpt-4o-mini")
 
chain = LLMChain(
    llm=llm,
    prompt=prompt
)
 
tasks = "Finish report, prepare slides, buy groceries"
result = chain.run(tasks)
 
print("Result:")
print(result)

summary_prompt = PromptTemplate(
    input_variables=["tasks"],
    template="Summarize the following tasks: {tasks}"
)
 
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    output_key="summary"
)
 

word_count_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Count the number of words in this text: {summary}"
)
 
word_count_chain = LLMChain(
    llm=llm,
    prompt=word_count_prompt,
    output_key="word_count"
)
 

classification_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Classify the topic of this text into one category (Work, Personal, Study): {summary}"
)
 
classification_chain = LLMChain(
    llm=llm,
    prompt=classification_prompt,
    output_key="category"
)
 

overall_chain = SequentialChain(
    chains=[summary_chain, word_count_chain, classification_chain],
    input_variables=["tasks"],
    output_variables=["summary", "word_count", "category"],
    verbose=True
)
 
result = overall_chain({
    "tasks": "Finish report, prepare slides, buy groceries, reply to emails"
})
 
print("\nFinal Result:")
print(result)

priority_prompt = PromptTemplate(
    input_variables=["tasks"],
    template="""
Assign a priority (High, Medium, Low) to each of the following tasks:
 
{tasks}
 
Return the answer in this format:
High: ...
Medium: ...
Low: ...
"""
)

priority_chain = LLMChain(
    llm=llm,
    prompt=priority_prompt,
    output_key="priority"
)

overall_chain = SequentialChain(
    chains=[
        summary_chain,
        word_count_chain,
        classification_chain,
        priority_chain
    ],
    input_variables=["tasks"],
    output_variables=["summary", "word_count", "category", "priority"],
    verbose=True
)
 
result = overall_chain({
    "tasks": """Finish report
Prepare slides
Buy groceries
Reply to emails"""
})
 
print(result)

