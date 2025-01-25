from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# Setup memory for conversation tracking
conversation_memory = ConversationBufferMemory()

# Main prompt
main_prompt = PromptTemplate(
    template=(
        "You are a conversational chatbot for a dental clinic.\n"
        "Use retrieved knowledge to guide users. Provide clear, helpful, and human-like explanations.\n\n"
        "Context: {context}\n\n"
        "### Conversation ###\n{conversation_history}\n"
        "Response:"
    ),
    input_variables=["context", "conversation_history"],
)

# Feedback collection prompt
feedback_prompt = PromptTemplate(
    template=(
        "Ask the user for their feedback in a friendly and polite manner after helping them.\n"
        "Encourage detailed and constructive feedback, and summarize what they shared."
    ),
    input_variables=[],
)

# Chains for conversation and feedback
conversation_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=main_prompt,
    memory=conversation_memory,
)
feedback_chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=feedback_prompt,
)

def chat_with_feedback(user_query: str):
    context = "Knowledge retrieved from RAG"
    response = conversation_chain.run(context=context, conversation_history=conversation_memory.load())
    print("Chatbot Response:", response)

    # Simulate feedback collection
    feedback_summary = feedback_chain.run()
    user_feedback = input("How was your experience? Provide feedback: ")
    print("Feedback Summary:", feedback_summary)

    return {"response": response, "user_feedback": user_feedback}
