from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import json
import re

load_dotenv()

# create model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# --- Types ---
class EvaluationResult(TypedDict):
    score: int
    comment: str

class EssayState(TypedDict):
    essay_content: str
    cot: EvaluationResult
    doa: EvaluationResult
    language: EvaluationResult
    summarized_eval: str
    average_score: float

# --- Helpers ---
def parse_json_safe(raw: str) -> Dict[str, Any]:
    """
    Cleans and parses raw model output into JSON.
    Falls back to {"score": 0, "comment": "Invalid response"} if parsing fails.
    """
    # Remove code fences if Gemini wraps the response
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"score": 0, "comment": "Invalid response"}

def evaluate_aspect(content: str, aspect: str, aspect_desc: str) -> EvaluationResult:
    """
    Generic evaluator for essay aspects (clarity, depth, language).
    """
    prompt = f"""
    You are an evaluator. Analyze the essay below for *{aspect_desc}*.

    Essay:
    {content}

    Return the result strictly in valid JSON only, with this schema:

    {{
      "score": <integer between 0 and 10>,
      "comment": "<your comments on {aspect_desc} in the essay>"
    }}
    """
    raw = model.invoke(prompt).content
    return parse_json_safe(raw)

# --- Nodes ---
def calculate_cot(state: EssayState):
    return {"cot": evaluate_aspect(state["essay_content"], "Clarity of Thought", "Clarity of Thought")}

def calculate_doa(state: EssayState):
    return {"doa": evaluate_aspect(state["essay_content"], "Depth of Analysis", "Depth of Analysis")}

def calculate_language_sc(state: EssayState):
    return {"language": evaluate_aspect(state["essay_content"], "Language", "use of Language")}

def final_evaluation(state: EssayState):
    cot, doa, language = state["cot"], state["doa"], state["language"]

    prompt = f"""
    Write a concise summary that combines the following evaluations into a single paragraph:
    - Clarity of Thought: {cot['comment']}
    - Depth of Analysis: {doa['comment']}
    - Language: {language['comment']}
    """
    summarized_eval = model.invoke(prompt).content
    average_score = (cot["score"] + doa["score"] + language["score"]) / 3
    return {"summarized_eval": summarized_eval, "average_score": average_score}

# --- Graph setup ---
graph = StateGraph(EssayState)

graph.add_node("calculate_cot", calculate_cot)
graph.add_node("calculate_doa", calculate_doa)
graph.add_node("calculate_language_sc", calculate_language_sc)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "calculate_cot")
graph.add_edge(START, "calculate_doa")
graph.add_edge(START, "calculate_language_sc")

graph.add_edge("calculate_cot", "final_evaluation")
graph.add_edge("calculate_doa", "final_evaluation")
graph.add_edge("calculate_language_sc", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# --- Run workflow ---
with open("essay.txt", "r") as f:
    essay = f.read()

initial_state = {"essay_content": essay}

final_state = workflow.invoke(initial_state)

print("Average Score:", final_state['average_score'])
print("Summary:\n", final_state['summarized_eval'])
