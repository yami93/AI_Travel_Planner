# workflow.py
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage
from agents import AgentState, MAX_ITERATIONS, call_planner, call_reviewer
from tools import google_search_tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

def call_tool_executor(state: AgentState) -> AgentState:
    print(f"\nLOG: --- Entering tool_executor node (Iteration {state['iteration']}) ---")
    last_message = state["chat_history"][-1]
    if not isinstance(last_message, AIMessage) or not getattr(last_message, "tool_calls", None):
        print("LOG: No tool call found in the last AIMessage.")
        return state

    tool_outputs = []
    for tool_call in last_message.tool_calls:
        print(f"LOG: Attempting to execute tool call: {json.dumps(tool_call, indent=2)}")
        tool_function = globals().get(tool_call['name'])
        if not tool_function or not callable(tool_function):
            output = f"Error: Tool function '{tool_call['name']}' not found or not callable."
            print(f"LOG: Error - {output}")
        else:
            try:
                query = tool_call['args'].get('query', '')
                if not query:
                    output = "Error: Missing 'query' argument for the tool."
                else:
                    output = tool_function(query)
            except Exception as e:
                output = f"Error executing tool {tool_call['name']}: {e}"
                print(f"LOG: Error - {output}")
        tool_outputs.append(ToolMessage(content=output, tool_call_id=tool_call['id'], name=tool_call['name']))

    print("LOG: Tool execution finished. Adding results to chat history.")
    return {
        "chat_history": state["chat_history"] + tool_outputs,
        "iteration": state["iteration"] + 1
    }

def decide_next_step(state: AgentState) -> str:
    print("\nLOG: --- Deciding next step ---")
    print(f"LOG: Current iteration: {state['iteration']}")
    if state['iteration'] > MAX_ITERATIONS:
        print("LOG: Max iterations reached. Ending graph.")
        return END

    last_message = state["chat_history"][-1]
    print(f"LOG: Last message type: {type(last_message).__name__}")

    if state.get("final_answer"):
        print("LOG: A final answer has been provided. Ending graph.")
        return END
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        print("LOG: Planner requested a tool call. Transitioning to tool_executor.")
        return "tool_executor"
    if isinstance(last_message, ToolMessage):
        print("LOG: Tool execution finished. Returning to planner to continue planning.")
        return "planner"
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
        print("LOG: Planner generated a response without a tool call. Transitioning to reviewer.")
        return "reviewer"

    print("LOG: No clear transition condition met. Defaulting to planner.")
    return "planner"

# --- Build and Compile the LangGraph Workflow ---
print("LOG: Building and compiling the LangGraph workflow...")
workflow = StateGraph(AgentState)
workflow.add_node("planner", call_planner)
workflow.add_node("tool_executor", call_tool_executor)
workflow.add_node("reviewer", call_reviewer)
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", decide_next_step)
workflow.add_conditional_edges("tool_executor", decide_next_step)
workflow.add_conditional_edges("reviewer", decide_next_step)
langgraph_app = workflow.compile()
print("LOG: Workflow compiled successfully.")