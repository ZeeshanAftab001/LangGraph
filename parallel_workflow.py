from langgraph.graph import StateGraph,START,END
from typing import TypedDict


# define the state
class PlayerState(TypedDict):

    runs: int
    balls: int
    fours: int
    sixes: int

    sr: float
    bpb: float
    boundary_percent: float
    summary: str

# We will pass partial state inorder to avoid error
# in Parallel Work Flows.Its can also be used in 
# sequential work flows.

# create a graph
graph=StateGraph(PlayerState)

# functions
def calculate_sr(state : PlayerState):
    sr=(state["runs"]/state["balls"])*100
    return {"sr":sr} # returning a partial state
    
def calculate_bpb(state : PlayerState):
    bpb = state['balls']/(state['fours'] + state['sixes'])
    return {'bpb': bpb}

def calculate_bp(state : PlayerState):
    boundary_percent = (((state['fours'] * 4) + (state['sixes'] * 6))/state['runs'])*100
    return {'boundary_percent': boundary_percent}

def summary(state:PlayerState):
   summary=f"""
    Strike Rate - {state['sr']} \n
    Balls per boundary - {state['bpb']} \n
    Boundary percent - {state['boundary_percent']}
    """
   return {"summary":summary}

# add nodes
graph.add_node("calculate_sr",calculate_sr)
graph.add_node("calculate_bpb",calculate_bpb)
graph.add_node("calculate_bp",calculate_bp)
graph.add_node("summary",summary)

# add edges
graph.add_edge(START,"calculate_sr")
graph.add_edge(START,"calculate_bpb")
graph.add_edge(START,"calculate_bp")

graph.add_edge("calculate_sr","summary")
graph.add_edge("calculate_bpb","summary")
graph.add_edge("calculate_bp","summary")

graph.add_edge("summary",END)


# compile graph
workflow=graph.compile()

# execute
initial_state={
    "runs": 100,
    "balls": 50,
    "fours": 6,
    "sixes": 5,
    }
final_state=workflow.invoke(initial_state)

print(final_state["summary"])