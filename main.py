# main.py
import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage
from agents import AgentState
from workflow import langgraph_app
from ui import display_ui, display_results

def generate_travel_plan(user_prompt: str) -> str:
    print("\nLOG: --- Starting travel plan generation ---")
    initial_state: AgentState = {
        "input": user_prompt,
        "chat_history": [HumanMessage(content=user_prompt)],
        "iteration": 0,
        "final_answer": ""
    }
    final_state = langgraph_app.invoke(initial_state)
    print("\nLOG: --- Travel plan generation complete ---")
    return final_state.get("final_answer", "Could not generate a plan. Please try again.")

def main_streamlit():
    destination, duration, travelers, activities = display_ui()

    if st.button("Generate Travel Plan"):
        if destination:
            user_input = f"A {duration}-day trip to {destination} for {travelers or 'unspecified group'}. "
            user_input += f"Interests: {activities or 'general sightseeing'}."

            with st.spinner("Creating your custom itinerary..."):
                plan = generate_travel_plan(user_input)
                display_results(plan, destination)
        else:
            st.warning("Please enter at least a destination to get started!")

if __name__ == "__main__":
    main_streamlit()