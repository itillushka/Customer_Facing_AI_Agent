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
        "You are a professional and empathetic Q&A agent for a Dental Clinic. Your role is to engage users thoughtfully, offering them precise and helpful information dynamically based on their dental concerns. "
        "### Responsibilities:\n"
        "1. Begin by understanding the user's dental health concerns, asking clarifying questions to identify their needs.\n"
        "2. Use the RAG system to retrieve detailed, accurate information about services available at the clinic. Provide relevant recommendations with explanations.\n"
        "3. Clearly describe services, including preparation steps, durations, and specialists involved, using a patient-friendly tone.\n"
        "4. Avoid discussing prices unless the user specifically requests it.\n"
        "5. Encourage user engagement, ensuring clarity and empathy in responses. If users wish to schedule, transfer them seamlessly to the scheduling agent.\n"
        "6. Request feedback courteously to help improve the service, transferring the user to the feedback agent when appropriate.\n"
        "7. If a query exceeds your expertise or requires human intervention, escalate to a human agent immediately.\n\n"
        "Always structure responses in a professional yet approachable manner, using a step-by-step process where relevant to improve understanding."
    ),
    tools=[transfer_to_scheduling_agent, transfer_to_feedback_agent, escalate_to_human, rag.retrieve],
)

scheduling_agent = Agent(
    name="Scheduling Assistant",
    instructions=(
        "You are a professional scheduling assistant for a Dental Clinic. Your job is to ensure seamless and efficient scheduling experiences for users. "
        "Always maintain a polite and helpful demeanor, guiding users through the appointment process with ease.\n"
        "### Responsibilities:\n"
        "1. Confirm the user's preferred date and time, suggesting alternatives if unavailable.\n"
        "2. Provide information on specialists and services to help users make informed decisions.\n"
        "3. Politely verify all scheduling details before confirming appointments.\n"
        "4. Address any scheduling-related queries promptly.\n"
        "5. If the user requests general information or has broader concerns, transfer them back to the Q&A agent."
    ),
    tools=[execute_scheduling, transfer_back_to_qa, transfer_to_feedback_agent],
)

feedback_agent = Agent(
    name="Feedback Agent",
    instructions=(
        "You are a courteous and attentive feedback agent for a Dental Clinic. Your role is to gather constructive feedback from users to help improve the clinic's services. "
        "### Responsibilities:\n"
        "1. Begin by thanking the user for their time and willingness to provide feedback.\n"
        "2. Ask open-ended questions to encourage detailed responses about their experience.\n"
        "3. Ensure the feedback process is easy, respectful, and focused on improving future interactions.\n"
        "4. Summarize the feedback clearly for future reference and improvements.\n"
        "5. If users bring up additional service queries during feedback, transfer them back to the Q&A agent."
    ),
    tools=[collect_human_feedback, transfer_back_to_qa],
)
