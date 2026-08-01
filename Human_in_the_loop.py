import os
from typing import TypedDict,Annotated,Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langgraph.types import interrupt,Command
from langgraph.checkpoint.memory import MemorySaver
load_dotenv()

#tools
search_tool=TavilySearch(max_results=3)
tools=[search_tool]
#writer
writer_llm=ChatMistralAI(
 model="mistral-small-latest",
 temperature=0.8
)
writer_llm_with_tools=writer_llm.bind_tools(tools)

#state building
class State(TypedDict):
 topic:str
 messages:Annotated[list,add_messages]
 draft:str
 review_feedback:str
 is_approved:bool
 attempt:int

#nodes
WRITER_SYSTEM_PROMPT=(
 "You are an expert LinkedIn content writer,Your job is to write "
 "engaging ,professional LinkedIn posts about the given topic."
 "If the topic requires up-to-date information ,statistics,or"
 "current trends, use the web-search tool to gather fresh context"
 "before writing. If you have already received feedback on a "
 "previous draft,carefully address every point in the new draft."
 "Rules for good LinkedIn posts:strong hook in the first line,"
 "1 clear takeway, easy to skim (short paragraphs),around"
 "150-200 words, ends with a question or call-to-action to invite "
 "engagement .Do not use hashtags"
)
WRITER_SYSTEM_PROMPT += (
  " Return ONLY the LinkedIn post. "
  "Do not explain your work. "
  "Do not say 'Here's the draft'. "
  "Do not include markdown separators."
)
def writer_node(state:State)->dict:
 """Writes (or rewrites )the LinkedIn posts.Can call Tavily to search first."""
 topic = state["topic"]
 previous_feedback = state.get("review_feedback", "")
 attempt=state.get("attempt",0)
 if attempt==0:
  user_message=(
   f"Write a LinkedIn post on this topic: '{topic}'. "
   f"if you need current information, search the web first"
   f"Return only the final LinkedIn post."
  )
 else:
  user_message = (
  f"Topic: {topic}\n\n"
  f"Previous Draft:\n{state['draft']}\n\n"
  f"Reviewer Feedback:\n{previous_feedback}\n\n"
  "Rewrite the LinkedIn post."
  "Fix every issue."
  "Keep the same topic."
  "Return only the final LinkedIn post."
)
 messages=[("system",WRITER_SYSTEM_PROMPT),("human",user_message)]
 response=writer_llm_with_tools.invoke(messages)

 return {
  "messages":[("human",user_message),response],
  "attempt":attempt
 }

tool_node=ToolNode(tools)

def extract_draft_node(state:State)->dict:
 """After the writer finishes tool calls, pulls the final text out as the draft.."""
 last_message=state['messages'][-1]
 draft=last_message.content.strip()
 print(f"\n\n Generated posts: \n{draft}\n")
 return {"draft":draft,
        "attempt": state["attempt"]+1
        }

def human_review_node(state:State)->dict:
 """Pauses the graph and waits for the human to approve or give feedback."""
 human_response=interrupt({
  "draft":state["draft"],
  "attempt":state["attempt"],
  "instruction":"Type 'approved' to accept, or type your feedback to request a rewrite."
 })
 response=human_response.strip()
 if response.lower() in ["approved","approve","yes","ok","good"]:
  return{
   "is_approved":True,
   "review_feedback":"Approved by human."
  }
 else:
  return{
   "is_approved":False,
   "review_feedback":response
  }

#router fuction
def should_use_tool(state:State):
 last_message=state['messages'][-1]
 if getattr(last_message,'tool_calls',None):
  return "tools"
 return "extract_draft"
def should_stop_looping(state:State):
 if state['is_approved']:
  print("post has been approved \n")
  return END
 if state['attempt']>=3:
  print("reached max attempts")
  return END
 return "writer"

#build the graph
graph=StateGraph(State)
graph.add_node("writer",writer_node)
graph.add_node("tools",tool_node)
graph.add_node("extract_draft",extract_draft_node)
graph.add_node("human_review",human_review_node)

graph.add_edge(START,"writer")
graph.add_conditional_edges(
 "writer",should_use_tool
)
graph.add_edge("tools","writer")
graph.add_edge("extract_draft","human_review")
graph.add_conditional_edges(
 "human_review",should_stop_looping
)
checkpointer=MemorySaver()
app=graph.compile(checkpointer=checkpointer)
print("="*55)
print("Welcome to the Linkedin Post Generator")
print("="*55)
print("\nThis tool will draft a LinkedIn post for you,review it")
print("itself,and iterate until it's publish-ready.")
print("="*55)
topic=input("\nWhat topic do you want a LinkedIn post about?\n").strip()
if not topic:
 print("\nNo topic given.Exiting.")
else:
 print("\nStarting generation...\n")
 config={"configurable":{"thread_id":"linkedin_session_1"}}
 initial_state={
  "topic":topic,
  "messages":[],
  "draft":"",
  "review_feedback":"",
  "is_approved":False,
  "attempt":0,
 }
 result=app.invoke(initial_state,config=config)
 while "__interrupt__" in result:
  interrupt_data = result["__interrupt__"][0].value
  print("="*55)
  print(f"DRAFT FOR YOUR REVIEW (Attempt {interrupt_data['attempt']})")
  print("="*55)
  print(interrupt_data["draft"])
  human_input=input("\nYour response: ").strip()
  result=app.invoke(
   Command(resume=human_input),
   config=config)
 print("\n"+"="*55)
 print("FINAL LINKEDIN POST")
 print("="*55)
 print(result["draft"])
 print(f"Total attempts: {result['attempt']}")
 print(f"Approved by human: {result['is_approved']}")


 






