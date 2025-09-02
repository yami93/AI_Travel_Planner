# agents.py
import json
from typing import List, Union, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from tools import google_search_tool
from langchain_core.tools import tool

# --- Define the Agent State for LangGraph ---
class AgentState(TypedDict):
    input: str
    chat_history: List[Union[HumanMessage, AIMessage, ToolMessage]]
    iteration: int
    final_answer: str

# --- Define the LLM ---
llm = ChatOllama(model='mistral', temperature=0.3)
llm_with_tools = llm.bind_tools([google_search_tool])
MAX_ITERATIONS = 5

def call_planner(state: AgentState) -> AgentState:
    print(f"\nLOG: --- Entering planner node (Iteration {state['iteration']}) ---")
    last_message = state["chat_history"][-1]
    prompt_modifier = ""
    if isinstance(last_message, AIMessage) and "Feedback:" in last_message.content:
        feedback = last_message.content.replace("Feedback: ", "")
        prompt_modifier = f"IMPORTANT: The previous plan was rejected. Please address the following issues in your new plan: {feedback}. Revise the itinerary to correct these errors."

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert travel planner. Generate a detailed day-by-day itinerary. "
                   "You may be given feedback if a previous plan was rejected. "
                   "Your final output should be a detailed, multi-day itinerary. If you need to search for information, call the google_search_tool with a specific query."
                   "\n\nFormat strictly as:\n"
                   "Day X: [Title]\n"
                   "- Morning: [Activity + location + meal suggestion]\n"
                   "- Afternoon: [Activity + location + meal suggestion]\n"
                   "- Evening: [Activity + location + meal suggestion]\n"
                   "\nRules:\n"
                   "1. Only include real, well-known attractions, restaurants, or landmarks.\n"
                   "2. If you are not certain about a place, use the google_search_tool to verify it.\n"
                   "3. Keep timing natural (e.g., 'late morning', 'around 7 PM') instead of rigid slots.\n"
                   "4. Ensure each day has morning, afternoon, and evening sections, with activities and meal suggestions included."),
        ("human", "{input}\n{prompt_modifier}")
    ])

    chain = prompt | llm_with_tools
    response = chain.invoke({"input": state["input"], "prompt_modifier": prompt_modifier})
    print("LOG: Planner finished.")
    return {
        "chat_history": state["chat_history"] + [response],
        "iteration": state["iteration"] + 1
    }

def call_reviewer(state: AgentState) -> AgentState:
    print(f"\nLOG: --- Entering reviewer node (Iteration {state['iteration']}) ---")
    last_message_content = state["chat_history"][-1].content
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a travel plan reviewer. Your task is to check the provided itinerary for completeness and correctness. "
                   "Identify any missing or incorrect details. If the itinerary is perfect, respond with '{{ \"status\": \"FINAL_ANSWER\" }}'. "
                   "If it needs work, respond with a JSON object strictly in the following format:\n"
                   "{{ \"status\": \"NEEDS_WORK\", \"feedback\": [\"Day 1: Missing afternoon activity\", \"Day 3: Inaccurate restaurant name\"] }}"
                   "The feedback should be a list of actionable issues. If no issues are found, the status must be 'FINAL_ANSWER'."),
        ("human", "Review this itinerary:\n\n{itinerary_content}")
    ])

    chain = prompt | llm
    response = chain.invoke({"itinerary_content": last_message_content})
    content = response.content.strip()

    try:
        feedback_json = json.loads(content)
        status = feedback_json.get("status")
        if status == "FINAL_ANSWER":
            print("LOG: Reviewer approved the plan as final.")
            return {
                "final_answer": last_message_content,
                "chat_history": state["chat_history"] + [response],
                "iteration": state["iteration"] + 1
            }
        elif status == "NEEDS_WORK":
            feedback_list = feedback_json.get("feedback", [])
            print(f"LOG: Reviewer flagged plan as needing work. Feedback: {feedback_list}")
            feedback_message = AIMessage(content=f"Feedback: {', '.join(feedback_list)}")
            return {
                "chat_history": state["chat_history"] + [response, feedback_message],
                "iteration": state["iteration"] + 1
            }
        else:
            raise ValueError("Invalid status field in JSON feedback.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"LOG: Reviewer returned invalid JSON. Error: {e}")
        return {
            "final_answer": last_message_content + "\n\n[⚠️ Reviewer returned invalid feedback, stopping.]",
            "chat_history": state["chat_history"] + [response],
            "iteration": state["iteration"] + 1
        }