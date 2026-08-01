import os
from langchain_groq import ChatGroq
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
load_dotenv()
llm=ChatGroq(
 model="llama-3.3-70b-versatile",
 temperature=0.1
)
def merge_score_dicts(existing:dict,newupdate:dict)->dict:
 if existing is None:
  return newupdate
 return{**existing,**newupdate}

#Create a state
class AnalyzerState(TypedDict):
 raw_text:str
 safety_score:Annotated[dict[str,int],merge_score_dicts]

#nodes
def toxcity_node(state:AnalyzerState)->dict:
 print("\n[Branch1]Analyzing toxcity and Hate Speech...")
 prompt=(
  "Analyze the following text for profanity,aggression,hate speech or toxcity"
  "Provide a score from 0 to 100,where 0 means perfectly clean and 100 means high toxcity"
  "Return ONLY the plain integer number,nothing else.\n\n"
  f"Text:\n{state['raw_text']}"
 )
 response=llm.invoke(prompt)
 print("LLM:", response.content)
 try:
  score=int(response.content.strip())
 except ValueError:
  score=0
 print("Returning:", {"safety_score": {"toxcity": score}})
 return {"safety_score": {"toxcity": score}}


def copyright_node(state:AnalyzerState)->dict:
 print("\n[Branch 2] Analyzing Copyright & Originality Risks...")
 prompt=(
  "Analyze the following text.Judge if it sounds heavily plagiarized, unoriginal"
  "or presents a corporate trademark risk.Provide a score from 0 to 100,"
  "where 0 means entirely original and 100 means high risk."
  "Return ONLY the plain integer number,nothing else.\n\n"
  f"Text:\n{state['raw_text']}"
 )
 response=llm.invoke(prompt)
 try:
  score=int(response.content.strip())
 except ValueError:
  score=0
  #return a sub-dictionary under the EXACT SAME state key
 return {"safety_score":{"copyright_risk":score}}

def culture_node(state:AnalyzerState)->dict:
  print("\n[Branch 3] Analyzing Regional and Cultural Sensitivity...")
  prompt=(
   "Analyze the following text for regional sensitivities,political landmines,"
   "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100"
   "where 0 means completely safe and 100 means highly offensive."
   "Return ONLY the plain integer number,nothing else.\n\n"
   f"\n{state['raw_text']}"
  )
  response =llm.invoke(prompt)
  try:
   score=int (response.content.strip())
  except ValueError:
   score=0
   #Return a sub-dictionary under the EXACT SAME state key
  return {"safety_score":{"cultural_insensitivity":score}}

builder=StateGraph(AnalyzerState)
builder.add_node("toxcity_node",toxcity_node)
builder.add_node("copyright_node",copyright_node)
builder.add_node("culture_node",culture_node) 

builder.add_edge(START,"toxcity_node")
builder.add_edge(START,"copyright_node")
builder.add_edge(START,"culture_node")

builder.add_edge("toxcity_node",END)
builder.add_edge("copyright_node",END)
builder.add_edge("culture_node",END)

app=builder.compile()

sample_script="""
Yo guys! Welcome back to the stream. Today I am going to show you how to hack into your friend's system using a script I copied from an online forum.
Honestly,tradional security protocols are absolute garbage and anyone still using them is an absolute idiot.Let's drive into the code! """

initial_state={
 "raw_text":sample_script,
 "safety_score":{} #initialized as an empty dictinary
}
final_state=app.invoke(initial_state)
print(final_state["safety_score"])