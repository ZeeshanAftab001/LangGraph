from langgraph.graph import StateGraph,END,START
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,List,Annotated
from langgraph.checkpoint.memory import InMemorySaver 
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage],add_messages]


graph=StateGraph(ChatState)

def chat(state:ChatState):

    messages = state['messages']
    response=model.invoke(messages).content
    return {"messages":[AIMessage(content=response)]}

graph.add_node("chat",chat)

graph.add_edge(START,"chat")
graph.add_edge("chat",END)


checkpointer=InMemorySaver()

workflow=graph.compile(checkpointer=checkpointer)

config={"configurable": {"thread_id": "1"}}

while(True):
    message=input("user>>")
    if message not in ["exit","close","quit","done"]:
        if message:
            response=workflow.invoke({
                "messages":HumanMessage(content=message)},config=config
            )
            print("ai>>",response["messages"][-1].content)
    else:
        break


print(workflow.get_state({"configurable":{"thread_id":1}}))