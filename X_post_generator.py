from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class TweetEvaluation(BaseModel):
    
    evaluation:Literal["approved","needs improvement"]=Field(...,description="Final evaluation result.")
    feedback:str=Field(...,description="feedback of the tweet.")

evaluation_model=model.with_structured_output(TweetEvaluation)

class TweetState(TypedDict):

    topic : str
    content:str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int


graph=StateGraph(TweetState)

# functions

def generate(state:TweetState):
    messages=[
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
        Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

        Rules:
        - Do NOT use question-answer format.
        - Max 280 characters.
        - Use observational humor, irony, sarcasm, or cultural references.
        - Think in meme logic, punchlines, or relatable takes.
        - Use simple, day to day english
    """)]
    content=model.invoke(messages).content
    return {"content":content}

def evaluate(state:TweetState):
    content=state["content"]
    messages = [
    SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
    HumanMessage(content=f"""
        Evaluate the following tweet:

        Tweet: "{content}"

        Use the criteria below to evaluate the tweet:

        1. Originality – Is this fresh, or have you seen it a hundred times before?  
        2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
        3. Punchiness – Is it short, sharp, and scroll-stopping?  
        4. Virality Potential – Would people retweet or share it?  
        5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

        Auto-reject if:
        - It's written in question-answer format (e.g., "Why did..." or "What happens when...")
        - It exceeds 280 characters
        - It reads like a traditional setup-punchline joke
        - Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

        ### Respond ONLY in structured format:
        - evaluation: "approved" or "needs_improvement"  
        - feedback: One paragraph explaining the strengths and weaknesses 
        """)
        ]
    
    response=evaluation_model.invoke(messages)

    return { "evaluation":response.evaluation,"feedback":response.feedback}

def optimize(state:TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
        Improve the tweet based on this feedback:
        "{state['feedback']}"

        Topic: "{state["topic"]}"
        Original Tweet:
        {state["content"]}

        Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
        """)
    ]

    response = model.invoke(messages).content
    iteration = state['iteration'] + 1

    return {"content":response,"iteration": iteration}

def route_evaluation(state:TweetState):
    if state["evaluation"]=="approved" or state['iteration'] >= state['max_iteration']:
        return "approved"
    else:
        return "needs_improvement"

# nodes
graph.add_node("generate",generate)
graph.add_node("evaluate",evaluate)
graph.add_node("optimize",optimize)

# edges

graph.add_edge(START,"generate")
graph.add_edge("generate","evaluate")
graph.add_conditional_edges("evaluate",route_evaluation,{"approved":END,"needs_improvement":"optimize"})
graph.add_edge("generate","evaluate")

workflow = graph.compile()

initial_state={"topic":"Pakistani Politicians",
               "iteration": 1,
                "max_iteration": 5}
final_state=workflow.invoke(initial_state)

print(final_state["iteration"])
print(final_state["content"])