from dotenv import dotenv_values, load_dotenv
load_dotenv()
config = dotenv_values(".env")

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DeepLake

org = config["ACTIVELOOP_ORG"]

dataset_path = 'hub://'+org+'/data'

embeddings = OpenAIEmbeddings()

db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False)

query = input("Enter query:")

ans = qa({"query": query})

print(ans)
