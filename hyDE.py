
from langchain import hub
import os
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

from langchain.load import dumps, loads

# Setting the necessary environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Add accordingly
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ['USER_AGENT'] = 'my-langchain-app'


def query_output(question,retriever):

        # HyDE document genration
        template = """Please write a passage to answer the question
        Question: {question}
        Passage:"""

        prompt_hyde = ChatPromptTemplate.from_template(template)

        # Selecting the model
        model = OllamaLLM(model="phi4-mini")

        generate_docs_for_retrieval = (
            prompt_hyde | model | StrOutputParser() 
        )


        # Retrieve
        retrieval_chain = generate_docs_for_retrieval | retriever 

        retrieved_docs = retrieval_chain.invoke({"question":question})

        # The prompt template
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        final_rag_chain = (
            prompt
            | model
            | StrOutputParser()
        )

        return final_rag_chain.invoke({"context":retrieved_docs,"question":question})
