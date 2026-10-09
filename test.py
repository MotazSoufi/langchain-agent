from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()  # loads your API key from .env

model = ChatAnthropic(model="claude-haiku-5-5")  # Haiku = fast and cheap
response = model.invoke("Say hello in one sentence.")
print(response.content)