import chainlit as cl
from dotenv import load_dotenv
import os
import json
import inspect
from pydantic import BaseModel
from typing import Optional, List, Dict
from openai import AsyncOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

class Agent(BaseModel):
    name: str
    model: str = "gpt-4o-mini"
    instructions: str
    tools: List

class Response(BaseModel):
    agent: Agent
    messages: List[Dict[str, str]]

# Define your tools and other functions here
def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id

def execute_refund(item_id, reason="not provided"):
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"

def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    exit()

def execute_order(product, price: int):
    """Price should be in USD."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."

def transfer_to_sales_agent():
    """User for anything sales or buying related."""
    return sales_agent

def transfer_to_issues_and_repairs():
    """User for issues, repairs, or refunds."""
    return issues_and_repairs_agent

def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return triage_agent

def place_order(item_name):
    return "success"

def transfer_to_refunds():
    return refund_agent

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "But make your questions subtle and natural."
    ),
    tools=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)

sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        ""
    ),
    tools=[execute_order, transfer_back_to_triage],
)

issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a customer support agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. First, ask probing questions and understand the user's problem deeper.\n"
        " - unless the user has already provided a reason.\n"
        "2. Propose a fix (make one up).\n"
        "3. ONLY if not satesfied, offer a refund.\n"
        "4. If accepted, search for the ID and then execute refund."
        ""
    ),
    tools=[execute_refund, look_up_item, transfer_back_to_triage],
)

refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)

tools = [execute_refund, look_up_item]

async def run_full_turn(agent: Agent, messages: List[Dict[str, str]]) -> Response:
    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = await client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append({"role": message.role, "content": message.content})

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if isinstance(result, Agent):  # if agent transfer, update current agent
                current_agent = result
                result = f"Transferred to {current_agent.name}. Adopt persona immediately."

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    print("Final messages:", messages)
    return Response(agent=current_agent, messages=messages[num_init_messages:])

def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments

def function_to_schema(func) -> dict:
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
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}")
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
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

# Initialize the agent and messages
agent = triage_agent
messages = []

@cl.on_message
async def on_message(message: cl.Message):
    global agent, messages

    print("Received message:", message.content)
    print("Current agent:", agent.name)

    # Append the user message to the messages list
    messages.append({"role": "user", "content": message.content})

    # Run the full turn with the current agent and messages
    response = await run_full_turn(agent, messages)
    agent = response.agent
    print(f"Messages {response.messages}")
    messages.extend(response.messages)
    # Send the agent's response back to the user
    for msg in response.messages:
        print('msg:', msg)
        if msg["role"] == "assistant" and not msg["content"].startswith("transfer_"):
            await cl.Message(content=msg["content"]).send()

    print("Sent messages:", response.messages)