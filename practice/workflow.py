from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
    InputRequiredEvent, HumanResponseEvent

)
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


class JokeEvent(Event):
    joke: str

class ProgressEvent(Event):
    msg: str

class FeedbackEvent(Event):
    joke: str
    feedback: str

class JokeFlow(Workflow):
    llm = OpenAI(api_key=api_key)

    @step
    async def generate_joke(self, ev: StartEvent) -> JokeEvent:
        topic = ev.topic
        prompt = f"Write your best joke about {topic}."
        response = await self.llm.acomplete(prompt)
        return JokeEvent(joke=str(response))
    @step
    async def critique_joke(self, ev: FeedbackEvent) -> StopEvent:
        joke = ev.joke
        feedback = ev.feedback
        prompt = f"Critique the following joke: {joke} based on feedback: {feedback}"
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=str(response))
    
class FeedbackFlow(Workflow):
    @step
    async def wait_for_feedback(self, ev: JokeEvent) -> FeedbackEvent:
        # Simulate waiting for human feedback
        feedback = input(f"Generated joke: {ev.joke}. Please provide feedback: ")
        return FeedbackEvent(joke=ev.joke, feedback=feedback)

# ---- Run the workflow ----
async def main():
    joke_workflow = JokeFlow()
    feedback_workflow = FeedbackFlow()
    # Start the joke workflow
    joke_event = await joke_workflow.generate_joke(StartEvent(topic="chemistry"))
    # Wait for human feedback
    feedback_event = await feedback_workflow.wait_for_feedback(joke_event)
    # Resume joke workflow with feedback
    critique_result = await joke_workflow.critique_joke(feedback_event)
    print("Critique result:", critique_result.result)
# Run the main logic
import asyncio
asyncio.run(main())

