# ui.py
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO
import re

# ---------------- PDF Export Function ----------------
def export_pdf(plan_text: str, destination: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ItineraryTitle", fontSize=18, leading=22,
        textColor=colors.darkblue, spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name="DayHeading", fontSize=13, leading=16, spaceAfter=8, textColor=colors.darkred, bold=True
    ))
    styles.add(ParagraphStyle(
        name="ItineraryBody", fontSize=11, leading=14, spaceAfter=6
    ))

    content = []
    content.append(Paragraph(f"✈️ Travel Itinerary for {destination}", styles["ItineraryTitle"]))
    content.append(Spacer(1, 12))

    for line in plan_text.split("\n"):
        if line.strip():
            if re.match(r"^Day\s+\d+:", line.strip()):
                content.append(Paragraph(line.strip(), styles["DayHeading"]))
            else:
                content.append(Paragraph(line.strip(), styles["ItineraryBody"]))

    doc.build(content)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ---------------- Helper: Format for Streamlit ----------------
def format_itinerary(plan: str) -> str:
    formatted = plan
    formatted = re.sub(r"(Day\s+\d+:)", r"**\1**", formatted)
    formatted = re.sub(r"(Morning\s*\([^)]+\):)", r"**\1**", formatted)
    formatted = re.sub(r"(Afternoon\s*\([^)]+\):)", r"**\1**", formatted)
    formatted = re.sub(r"(Evening\s*\([^)]+\):)", r"**\1**", formatted)
    return formatted

def display_ui():
    st.set_page_config(page_title="AI Travel Planner", page_icon="✈️")
    st.title("✈️ AI Travel Planner")
    st.markdown("""
    Welcome to your personal AI travel agent!
    Fill in your trip details below, and I'll generate a detailed itinerary for you.
    """)

    st.subheader("Tell me about your trip ✍️")
    col1, col2 = st.columns(2)
    with col1:
        destination = st.text_input("Destination", placeholder="e.g., Paris, France")
        duration = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=3)
    with col2:
        travelers = st.text_input("Who’s traveling?", placeholder="e.g., Family of 4, Couple, Solo")

    activities = st.text_area(
        "Activities / Interests",
        placeholder="e.g., Art museums, local food, nature walks, shopping"
    )

    return destination, duration, travelers, activities

def display_results(plan: str, destination: str):
    st.markdown("---")
    st.subheader("Your Generated Itinerary")
    st.markdown(format_itinerary(plan))

    pdf_bytes = export_pdf(plan, destination or "Your Trip")
    st.download_button(
        label="📥 Download Itinerary as PDF",
        data=pdf_bytes,
        file_name=f"itinerary_{destination.replace(' ', '_')}.pdf",
        mime="application/pdf"
    )