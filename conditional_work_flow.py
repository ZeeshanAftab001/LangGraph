from langgraph.graph import StateGraph,END,START
from typing import TypedDict,Literal

class TicketState(TypedDict):
    destination:str
    bus:str
    fare:int
    summary:str

graph=StateGraph(TicketState)

def check_bus(state:TicketState):
    
    destination=state["destination"]
    if destination=="karachi":
        bus="bus1"
    elif destination=='lahore':
        bus="bus2"
    else:
        bus="bus3"
    return {"bus":bus}

def to_karachi(state:TicketState):
    fare=1000
    return {"fare":fare}
def to_lahore(state:TicketState):
    fare=1500
    return {"fare":fare}
def to_peshawar(state:TicketState):
    fare=1600
    return {"fare":fare}
def fare_summary(state:TicketState):
    summary=f'''
    The ticket of {state["destination"]} costs {state['fare']}.
    '''
    return {"summary":summary}

def get_bus(state:TicketState)->Literal["to_karachi","to_lahore","to_peshawar"]:

    if state["bus"]=="bus1":
        return "to_karachi"
    elif state["bus"]=="bus2":
        return "to_lahore"
    else:
       return "to_peshawar"

graph.add_node("check_bus",check_bus)
graph.add_node("to_karachi",to_karachi)
graph.add_node("to_lahore",to_lahore)
graph.add_node("to_peshawar",to_peshawar)
graph.add_node("fare_summary",fare_summary)

graph.add_edge(START,"check_bus")
graph.add_conditional_edges("check_bus",get_bus)
graph.add_edge("to_karachi","fare_summary")
graph.add_edge("to_lahore","fare_summary")
graph.add_edge("to_peshawar","fare_summary")

graph.add_edge("fare_summary",END)


workflow=graph.compile()


initial_state={"destination":"lahore"}
final_state=workflow.invoke(initial_state)

print(final_state['summary'])