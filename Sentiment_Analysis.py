from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field

load_dotenv()

class SentimentSchema(BaseModel):
    sentiment : Literal["postive","negetive"] = Field(description="sentiment of the review")

class DiagnosisSchema(BaseModel):
    issue_type:Literal["UX","Preformance","Bug","Support","Others"] = Field(description="The category of Issue mentioned in the review.")
    tone : Literal["anger","frustration","disappointed","calm"]  = Field(description="The emotional tone of the user.")
    urgency:Literal["low","medium","high"] = Field(description="How urgent or critical the issue appears to be.")


model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

model1=model.with_structured_output(SentimentSchema)
model2=model.with_structured_output(DiagnosisSchema)

class State(TypedDict):
    review:str
    sentiment:str
    diagnosis:dict
    response:str



graph=StateGraph(State)

# functions
def find_sentiment(state:State):
    review=state["review"]
    prompt=f'''
    What is the sentiment of the review below?
    {review}
    '''
    sentiment=model1.invoke(prompt).sentiment
    return {"sentiment":sentiment}

def check_sentiment(state:State) -> Literal["run_diagnosis","positive_response"]:
    if state["sentiment"]=="positive":
        return "positive_response"
    else:
        return "run_diagnosis"
    

def positive_response(state:State):
    review=state["review"]
    prompt=f"generate a response for this review.{review}"
    response=model.invoke(prompt).content
    return {"response":response}

def run_diagnosis(state:State):
    review=state["review"]
    prompt=f'''
    Diagnose this negetive review :\n\n{state["review"]}\n
    return issue type,tone and urgency.
    '''
    diagnosis=model2.invoke(prompt)
    return {"diagnosis":diagnosis.model_dump()}

def negitive_prompt(state:State):
    diagnosis=state["diagnosis"]
    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
    Write a short empathetic, helpful resolution message.
    """
    response=model.invoke(prompt).content
    return {"response":response}

# adding nodes
graph.add_node("find_sentiment",find_sentiment)
graph.add_node("run_diagnosis",run_diagnosis)
graph.add_node("negitive_prompt",negitive_prompt)
graph.add_node("positive_response",positive_response)

# adding edges
graph.add_edge(START,"find_sentiment")
graph.add_conditional_edges("find_sentiment",check_sentiment)
graph.add_edge("run_diagnosis","negitive_prompt")
graph.add_edge("negitive_prompt",END)
graph.add_edge("positive_response",END)


work_flow=graph.compile()

initial_state={
    "review":"The application is very very difficult to use!!!"
}

final_state=work_flow.invoke(initial_state)

print(final_state["response"])