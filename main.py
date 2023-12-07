from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from flask import Flask, request
import weaviate
import os 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Weaviate
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.streaming_stdout_final_only import FinalStreamingStdOutCallbackHandler

class Message(BaseModel):
    role: str
    content: str
    id: Optional[str] = None

class ChatPayload(BaseModel):
    messages: List[Message]

load_dotenv()
app = Flask(__name__)

def get_response_endpoint(chat_payload: ChatPayload, stream=False):

    auth_config = weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"])

    weaviate_client = weaviate.Client(
    url="https://cisco-lomxf30k.weaviate.network",
    auth_client_secret=auth_config
    )

    embeddings = OpenAIEmbeddings(disallowed_special=())
    db = Weaviate(weaviate_client, "LangChain_198d6a7a03af4920914bf5b129b32527", "text", embeddings)
    retriever = db.as_retriever(
        search_type="mmr",  # You can also experiment with "similarity"
        search_kwargs={"k": 8},
    )

    llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4", callbacks=[FinalStreamingStdOutCallbackHandler()])
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

    chat_history = []
    question = chat_payload.messages[-1].content
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result["answer"]

@app.post("/chat")
def chat():
    if request.method == "POST":
        chat_payload = request.get_json()
        chat_payload = ChatPayload(**chat_payload)
        return {"message" : get_response_endpoint(chat_payload, stream=False) }

if __name__ == "__main__":
    get_response_endpoint(ChatPayload(**{
    "messages": [
        {
            "id": "Dfbjjjw",
            "content": "What can I do to optimize my brain health?",
            "role": "user"
        }
    ]
}), stream=False)