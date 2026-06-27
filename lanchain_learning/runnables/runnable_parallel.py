from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

"""The Runnable Parallel is a class that allows you to run multiple runnables in parallel.
Each runnable takes the same input and produces its own output."""

prompt1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

"""In this example, we will create a Runnable Parallel that takes a topic as input,
and generates a tweet and a Linkedin post about that topic in parallel.Each runnable will use the same input and 
produce its own output.We have to name the runnables in the Runnable Parallel so that we can access their outputs later.
The Runnable Parallel will consist of the following steps:"""

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'linkedin': RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])
