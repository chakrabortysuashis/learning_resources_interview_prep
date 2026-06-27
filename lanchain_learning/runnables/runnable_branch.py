from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch,RunnableLambda

load_dotenv()

"""The Runnable Branch is a class that allows you to create a branch in a runnable sequence.It is 
like a if else statement in programming. The Runnable Branch takes a condition and two runnables as input."""

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

report_gen_chain = prompt1 | model | parser
"""RunnableBranch takes in multiple tuples of (condition, runnable) and evaluates the conditions in order.
The first condition that evaluates to True will have its corresponding runnable executed.The last
 runnable in the RunnableBranch is the default runnable that will be executed if none of the conditions are met.
In this example, we will create a Runnable Branch that takes a topic as input, generates a detailed report 
on that topic, and then checks the length of the report.
If the report is longer than 300 words, it will summarize the report.
 If the report is shorter than 300 words, it will return the report as is. 
 We will use the ChatOpenAI model to generate the report and summarize it.
The Runnable Branch will consist of the following steps:"""
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'Russia vs Ukraine'}))


