from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

class BMI(TypedDict):

    weigth_kg:float
    heigh_m:float
    bmi:float
    category:str
    


graph=StateGraph(BMI)


def calculate_bmi(state:BMI)->BMI:
    weight=state["weigth_kg"]
    height=state["heigh_m"]
    
    bmi=weight/(height**2)

    state["bmi"]=round(bmi,2)

    return state

def label_bmi(state:BMI)->BMI:
    bmi=state["bmi"]

    if bmi < 18:
        state["category"]="under weight"

    elif bmi > 18 and bmi<25:
        state["category"]="moderate bmi"

    else:
        state["category"]="over weight"
    
    return state




#add nodes

graph.add_node("calculate_bmi",calculate_bmi)
graph.add_node("label_bmi",label_bmi)

#add edges

graph.add_edge(START,"calculate_bmi")
graph.add_edge("calculate_bmi","label_bmi")
graph.add_edge("label_bmi",END)

#compile graph

workflow=graph.compile()

#execute graph
initial_state={"weigth_kg":80,"heigh_m":1.73}

final_state=workflow.invoke(initial_state)

print("BMI Category is : ",final_state["category"])
print("BMI is : ",final_state["bmi"])

# from IPython.display import Image, display
# display(Image(workflow.get_graph().draw_mermaid_png()))   