from pydantic import BaseModel
from typing import Optional
import json
import inspect
import os
from dotenv import load_dotenv
from agents import Agent, Response, qa_agent, scheduling_agent, feedback_agent
import chainlit as cl
from openai import AsyncOpenAI

load_dotenv()
cl.instrument_openai()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# Store messages and agent context globally for simplicity
agent = qa_agent
messages = []

async def run_full_turn(agent, messages):
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # Turn Python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. Get OpenAI completion ===
        response = await client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # Print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # If finished handling tool calls, break
            break

        # === 2. Handle tool calls ===
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if type(result) is Agent:  # If agent transfer, update current agent
                current_agent = result
                result = (
                    f"Transferred to {current_agent.name}. Adopt persona immediately."
                )

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # === 3. Return last agent used and new messages ===
    yield Response(agent=current_agent, messages=messages[num_init_messages:])


def execute_tool_call(tool_call, tools, agent_name):
    """Executes the corresponding tool function with its arguments."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}: {name}({args})")

    return tools[name](**args)


def function_to_schema(func):
    """Converts Python functions into LangChain tool schema."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {str(e)}")

    parameters = {}
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}

    required = [
        param.name for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

@cl.on_message
async def handle_message(message: cl.Message):
    """Chainlit message handler."""
    global agent, messages

    # Append user message to conversation
    messages.append({"role": "user", "content": message.content})

    # Run the interaction
    async for response in run_full_turn(agent, messages):
        print(response)
        # At this point, `response` is a Response object
        # Access the content directly from the response object
        final_response = response.messages[-1].content
        await cl.Message(content=f"{final_response}").send()

    # Update the global agent if transitioned
    agent_context = agent.name
    if agent_context == "Feedback Agent":
        # Explicitly collect feedback via the feedback agent
        feedback_result = feedback_agent.tools[0]()  # `collect_human_feedback`
        await cl.Message(content=f"Feedback collected: {feedback_result}").send()
