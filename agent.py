from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool
def multiply(a: float, b: float) -> float:
    # This is a docstring that gives the LLM a description of what the tool does.
    """Multiply two numbers together."""
    return a * b

@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

model = ChatAnthropic(model="claude-haiku-5-5")

agent = create_agent(
    model=model,
    tools=[multiply, get_current_time],
    system_prompt="You area a helpful assistant. Use tools when needed.",
)

# the conversation gets stored here so the ai agent can remember
messages = []

print("Chat with your agent! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # this add the user's message to the history
    messages.append({"role": "user", "content": user_input})

    # sends the whole history to the ai agent
    result = agent.invoke({"messages": messages})

    # save the updated history
    messages = result["messages"]

    print("Agent:", messages[-1].content, "\n")