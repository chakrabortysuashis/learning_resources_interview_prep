from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

"""The Runnable Sequence is a class that allows you to chain together multiple runnables in a sequence.
Each runnable takes the output of the previous runnable as its input. 
This is useful for creating complex workflows that require multiple steps.
In this example, we will create a Runnable Sequence that takes a topic as input,
generates a joke about that topic, and then explains the joke. 
We will use the ChatOpenAI model to generate the joke and explain it.
The Runnable Sequence will consist of the following steps:"""

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))