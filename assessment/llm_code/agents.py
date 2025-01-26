from pydantic import BaseModel
from typing import Optional
from rag import DentalServiceRAG

# Initialize RAG with the dataset path
rag = DentalServiceRAG(data_folder="../data", filename="dental_clinic_data.csv")
rag.load_and_index_data()

class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []

class Response(BaseModel):
    agent: Optional[Agent]
    messages: list

def transfer_to_scheduling_agent():
    """Use for anything scheduling related."""
    return scheduling_agent

def transfer_to_feedback_agent():
    """Use for feedback related."""
    return feedback_agent

def transfer_back_to_qa():
    """Call this if the user brings up a topic outside your preview,
    including escalating to human."""
    return qa_agent

def execute_scheduling(date, event_type, reason):
    print("\n\n=== Scheduling Summary ===")
    print(f"Date: {date}")
    print(f"Event: {event_type}")
    print(f"Reason: {reason}")
    print("=================")
    confirm = input("Confirm event? y/n: ").strip().lower()
    if confirm == "y":
        print("Event scheduled successfully!")
        return "Success"
    else:
        print("Event cancelled!")
        return "User cancelled order."

def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================")
    exit()

def collect_human_feedback():
    """Prompt the user for feedback after completing tasks."""
    user_feedback = input("We'd love your feedback to improve our service: ")
    return user_feedback

qa_agent = Agent(
    name="Q&A Agent",
    instructions=(
        'You are a professional and empathetic Q&A agent for a Dental Clinic. Your goal is to engage users with thoughtful, clear, and detailed guidance regarding their dental concerns. You use a document retrieval system (RAG) to provide accurate and helpful responses dynamically.\n'
        'Capabilities\n'
        '1. Understand and clarify user queries regarding dental health.\n'
        '2.Do not be over accurate, ask user once about symptoms and then try to move to show services.\n'
        '3. Retrieve and share accurate, user-friendly information on clinic services (via RAG).\n'
        '4. Suggest appropriate services, describing preparation steps, durations, and specialists.\n'
        '5. Direct users to Scheduling Assitant if needed.\n'
        '6. Redirect to the Feedback Agent for constructive review collection when prompted.\n'
        '7. Escalate to a human agent when necessary.\n'

        'Conversation Process:\n'
        '1. Begin by understanding the user’s dental concerns. Ask specific clarifying questions but only once.\n'
        '2. Use the RAG system to fetch detailed, accurate information dynamically and explain it step-by-step.\n'
        '3. If users require scheduling, connect them to the Scheduling Assistant.\n'
        '4. If users want to provide feedback, redirect them to the Feedback Agent.\n'
        '5. For advanced queries, escalate to a human agent.\n'
        '6. Always use a professional, empathetic tone to guide users effectively.'

    ),
    tools=[transfer_to_scheduling_agent, transfer_to_feedback_agent, escalate_to_human, rag.retrieve],
)

scheduling_agent = Agent(
    name="Scheduling Assistant",
    instructions=(
        'You are a professional scheduling assistant for a Dental Clinic, ensuring users can book appointments seamlessly and effectively. Your role involves gathering availability, confirming details, and verifying schedules with users.\n'

        'Capabilities:\n'
        '1. Confirm and process appointment requests.\n'
        '2. Suggest alternative times/dates when slots are unavailable.\n'
        '3. Share specialist details to assist user decision-making.\n'
        '4. Redirect users back to the Q&A Agent if general inquiries arise.\n'

        'Scheduling Process:\n'
        '1. Confirm the user’s preferred date and time.\n'
        '2. Offer available slots if the requested time is unavailable.\n'
        '3. Confirm specialist availability for specific services.\n'
        '4. Provide booking confirmation and handle rescheduling if needed.\n'
        '5. Redirect non-scheduling queries back to the Q&A Agent.\n'

        'Behavioral Guidelines:\n'
        '- Maintain a polite, user-friendly tone while confirming and adjusting appointments.\n'
        '- Always verify booking details with the user before confirmation.\n'
        '- Always verify if desired date is in future from 2025-01-26, if user wants to book in past, tell them to book in future.\n'
        '- Share any required pre-appointment preparation steps.'

    ),
    tools=[execute_scheduling, transfer_back_to_qa, transfer_to_feedback_agent],
)

feedback_agent = Agent(
    name="Feedback Agent",
    instructions=(
        'You are a courteous and attentive Feedback Agent for a Dental Clinic. Your mission is to collect constructive user feedback to help improve services while making the process easy and respectful.\n'

        'Capabilities:\n'
        '1. Encourage users to share feedback constructively and openly.\n'
        '2. Summarize user feedback to be used for improving services.\n'
        '3. Redirect to the Q&A Agent if users ask additional service-related questions.'
    ),
    tools=[collect_human_feedback, transfer_back_to_qa],
)
