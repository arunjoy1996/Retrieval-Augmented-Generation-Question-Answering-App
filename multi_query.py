

from langchain import hub

import os
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.load import dumps, loads
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Setting the necessary environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ['USER_AGENT'] = 'my-langchain-app'


def query_output(question,retriever):


        # Multi Query: Different Perspectives
        template = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. Original question: {question}"""
        
        prompt_perspectives = ChatPromptTemplate.from_template(template)


        # Selecting the model
        model = OllamaLLM(model="phi4-mini")

        generate_queries = (
            prompt_perspectives 
            | model
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
        )



        # Get unique documents
        def get_unique_union(documents: list[list]):
            """ Unique union of retrieved docs """
            # Flatten list of lists, and convert each Document to string
            flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
            
            unique_docs = list(set(flattened_docs))
            # Return
            return [loads(doc) for doc in unique_docs]

        
        # Retrieve
        retriever.search_kwargs["k"] = 1  # Change k or the number of documents retrieved dynamically

        retrieval_chain = generate_queries | retriever.map() | get_unique_union
        docs = retrieval_chain.invoke({"question":question})


        # Prompt template
        template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)


        final_rag_chain = (
            {"context": retrieval_chain, 
            "question": itemgetter("question")} 
            | prompt
            | model
            | StrOutputParser()
        )

        return final_rag_chain.invoke({"question":question})