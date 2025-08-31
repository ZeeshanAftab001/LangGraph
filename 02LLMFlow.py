from langgraph.graph import START,StateGraph,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7)

# create a state
class LLMState(TypedDict):

    question : str
    answer : str

# functions

def llmQuestion(state : LLMState) -> LLMState:

    # get question from the state
    question=state["question"]

    # create a prompt
    prompt=f"answer the following question,{question}"

    # invoke the model
    answer=model.invoke(prompt).content
    
    # update the state
    state["answer"]=answer

    # return the state
    return state



# create a graph

graph=StateGraph(LLMState)

# add nodes
graph.add_node("LLM_question",llmQuestion)


# add edges
graph.add_edge(START,"LLM_question")
graph.add_edge("LLM_question",END)

# compile the graph
workflow=graph.compile()

# execute the graph

initial_state={"question":"Who is the prime minister of pakistan?"}

final_state=workflow.invoke(initial_state)

print("Answer to your question is : ",final_state["answer"])

from IPython.display import Image, display
display(Image(workflow.get_graph().draw_mermaid_png()))   