from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv


load_dotenv()


# make model
model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7)

# define graph state
class LLMState(TypedDict):
    topic:str
    topic_outline:str
    content:str
    rating:str

# create graph
graph=StateGraph(LLMState)

# functions
def get_outline(state:LLMState)->LLMState:
    
    # get topic from state
    topic=state['topic']

    # create a prompt
    prompt=f"generate an detail outline for this topic {topic}"

    # invoke model
    outline=model.invoke(prompt).content

    # update state
    state['topic_outline']=outline

    # return state
    return state 

def get_blog(state:LLMState)->LLMState:

    # get topic from the state
    topic=state["topic"]

    # get outline from the state
    outline=state["topic_outline"]

    # create a prompt
    prompt=f'generate a blog of about 200 words on the topic {topic} by using this outline,{outline}'

    # invoke model
    content=model.invoke(prompt).content

    # update the state
    state["content"]=content

    # return the state
    return state

def evaluate_blog(state:LLMState)->LLMState:
    
    # get topic outline
    outline=state["topic_outline"]

    # get the blog content
    content=state["content"]

    # create a prompt
    prompt=f"On the basis of my topic outline which is {outline} please evaluate the blog which is {content}.Provide me a rating in between 0-10 in result."

    # invoke the model
    rating=model.invoke(prompt).content

    # update the state
    state["rating"]=rating

    # return the state
    return state

# create nodes
graph.add_node("get_outline",get_outline)
graph.add_node("get_blog",get_blog)
graph.add_node("evaluate_blog",evaluate_blog)

# create edges

graph.add_edge(START,"get_outline")
graph.add_edge("get_outline","get_blog")
graph.add_edge("get_blog","evaluate_blog")
graph.add_edge("evaluate_blog",END)

# compile graph

workflow=graph.compile()

# execute 

initial_state={"topic":"Dehumanization and AI"}
final_state=workflow.invoke(initial_state)


with open("Dehumanization & AI.txt","w") as f:
    f.write(f"The Rating is : {final_state["rating"]}")
    f.write(final_state["topic"])
    f.write(final_state["topic_outline"])
    f.write(final_state["content"])
