from dotenv import dotenv_values, load_dotenv

load_dotenv()
config = dotenv_values(".env")

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DeepLake

loader = TextLoader("messages.txt")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(pages)

print (texts)

org = config["ACTIVELOOP_ORG"]

dataset_path = 'hub://'+org+'/data'
embeddings = OpenAIEmbeddings()
db = DeepLake.from_documents(texts, embeddings, dataset_path=dataset_path)
