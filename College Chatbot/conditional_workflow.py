import os
from  typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()
#step 1-building retriever
embeddings=HuggingFaceEmbeddings(
 model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def build_retriever(pdf_path: str):
 loader=PyPDFLoader(pdf_path)
 document=loader.load()
 splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=100)
 chunks=splitter.split_documents(document)

 vectorstore=FAISS.from_documents(chunks,embeddings)
 return vectorstore.as_retriever(search_kwargs={"k":4})

academic_retriever = build_retriever(
    r"C:\Users\runal\Downloads\Agentic AI\College Chatbot\academics_handbook.pdf"
)

fee_retriever = build_retriever(
    r"C:\Users\runal\Downloads\Agentic AI\College Chatbot\fee_structure.pdf"
)

llm=ChatGroq(
 model="llama-3.3-70b-versatile",
 temperature=0.5
)

#step-2 state
class State(TypedDict):
 programme:str
 messages:Annotated[list,add_messages]
 query_type:str
 retriever_context:str

#step-3 nodes
def classifier_node(state:State)->dict:
 """Look at the latest user message and decide which path to take."""
 last_message=state['messages'][-1].content 
 prompt=(
  "Classify the following student query into exactly one category:"
  "'academic','fee',or 'general'.\n\n"
  "Use 'academic' for questions about attendance,exams,grading,credits,"
  "promotion, course structure,summer training,or degree requirements,\n"
  "Use 'fee' for questions about tuition,payment,refung ,late charges,"
  "scholarships, or any money-related topic.\n"
  "Use 'general' for greetings, casual talk, or anything not related to "
  "the college rules or fee.\n\n"
  f"Query:{last_message}\n\n"
  "Return only one word: academic,fee or general."
 )
 response=llm.invoke(prompt)
 category=response.content.strip().lower()

 if "academic" in category:
  category="academic"
 elif "fee" in category:
  category="fee"
 else:
  category="general"

 return {"query_type":category}

def academic_rag_node(state:State)->dict:
 """Retrievers relevant chunks from the academics handbook."""
 query=state["messages"][-1].content
 docs=academic_retriever.invoke(query)
 context="\n\n".join([doc.page_content for doc in docs])
 return {
  "retriever_context": context
}

def fee_rag_node(state:State)->dict:
 """Retrievers relevant chunks from the academics handbook."""
 query=state["messages"][-1].content
 docs=fee_retriever.invoke(query)
 context="\n\n".join([doc.page_content for doc in docs])
 return{
  "retriever_context":context
  }

def general_node(state:State)->dict:
 """Answer directly using the LLM's own knowledge,no retrieval needed."""
 return{
  "retriever_context": "NO_RETRIEVAL_NEEDED"
  }

def response_node(state:State)->dict:
 """Generates the final answer,personalised using the student's programme."""
 query=state["messages"][-1].content
 programme=state.get("programme","Unknown")
 context=state["retriever_context"]
 if context == "NO_RETRIEVAL_NEEDED":
  prompt=(
   f"You are a friendly college assistant talking to a{programme} student."
   f"Answer this question using your own general knowledge:\n\n{query}"
  )
 else:
  prompt=(
   f"You are a college assistant helping a {programme} student."
   f"Use the following context from the official college document and answer"
   f"the question accurately. If the context mentions specific are "
   f"different programmes,highlight the one relevant to {programme}"
   f"Context:\n{context}\n\n"
   f"Question:{query}\n\n"
  )
 response=llm.invoke(prompt)
 return {
   "messages": [
    AIMessage(content=response.content.strip())
   ]
  }

#Step 4 -router function
def router_query(state:State):
 if state['query_type']=='academic':
  return "academic_rag"
 elif state['query_type']=="fee":
  return "fee_rag"
 else:
  return "general"

#step 5 Building the graph
graph=StateGraph(State)
graph.add_node("classifier",classifier_node)
graph.add_node("academic_rag",academic_rag_node)
graph.add_node("fee_rag",fee_rag_node)
graph.add_node("general",general_node)
graph.add_node("response",response_node)

#edges
graph.add_edge(START,"classifier")
graph.add_conditional_edges(
 "classifier",router_query
)
graph.add_edge("academic_rag","response")
graph.add_edge("fee_rag","response")
graph.add_edge("general","response")
graph.add_edge("response",END)
app=graph.compile()

#step 6 run the code
import streamlit as st


st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 College Assistant")
st.write("Ask questions about academics, fees, or general college information.")

# -----------------------------
# Programme Selection
# -----------------------------
if "programme" not in st.session_state:
    st.session_state.programme = "BCA"

programme = st.selectbox(
    "Select Your Programme",
    ["BCA", "BBA", "B.Com(H)"],
    index=["BCA", "BBA", "B.Com(H)"].index(st.session_state.programme)
)

st.session_state.programme = programme

# -----------------------------
# Chat History
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat Input
# -----------------------------
user_query = st.chat_input("Ask your question...")

if user_query:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Run LangGraph
    result = app.invoke(
        {
            "programme": st.session_state.programme,
            "messages": [("human", user_query)]
        }
    )

    answer = result["messages"][-1].content

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )