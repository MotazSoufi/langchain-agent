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

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 23.5 times 48? Also, what time is it?"}]}
)

for message in result["messages"]:
    message.pretty_print()