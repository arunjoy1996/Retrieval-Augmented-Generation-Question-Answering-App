

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableLambda

# Setting the necessary environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Add accordingly
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ['USER_AGENT'] = 'my-langchain-app'


def query_output(question,retriever):

        # Prompt
        prompt = hub.pull("rlm/rag-prompt")

        # LLM
        model = OllamaLLM(model="phi4-mini")

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)



        # Wrap the function
        format_docs_runnable = RunnableLambda(format_docs)
        
        # The pipeline
        rag_chain = (
            {
                "context": retriever | format_docs_runnable,
                "question": RunnablePassthrough()
            }
            | prompt
            | model
            | StrOutputParser()
        )

        return rag_chain.invoke(question)
