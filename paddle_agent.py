from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

timeSlots = ["10:00", "11:00", "12:00"]
courts = ["Court A", "Court B", "Court C"]
bookings = [
    {"id": 1, "name": "Motaz", "court": "Court B", "date":"20-10-2026", "time": "10:00"}
]

def is_valid_date(date: str) -> bool:
    try:
        datetime.strptime(date, "%d-%m-%Y").strftime("%A")
        return True
    except ValueError:
        return False

@tool
def get_today_date() -> str:
    """Get today's date."""
    return datetime.now().strftime("%d-%m-%Y (%A)")

@tool
def check_availability(date: str) -> str:
    """Check which courts and time slots are free."""
    if not is_valid_date(date):
        return "Invalid date. Use format DD-MM-YYYY."

    availableSlots = []
    for court in courts:
        for timeSlot in timeSlots:
            taken = False
            for booking in bookings:
                if booking["court"] == court and booking["time"] == timeSlot and booking["date"] == date:
                    taken = True
                    break
            if not taken:
                availableSlots.append(f"{date} at {timeSlot} in {court}")

    return "\n".join(availableSlots)
            
@tool
def booked():
    """View who booked courts"""
    return bookings

@tool
def book_court(name: str, court: str, date: str, timeSlot: str) -> str:
    """Book a court"""
    if court not in courts:
        return f"Unkown court. Options: {', '.join(courts)}"
    if timeSlot not in timeSlots:
        return f"Invalid time. Options: {', '.join(timeSlots)}"
    if not is_valid_date(date):
        return "Invalid date. Use format DD-MM-YYYY"

    for b in bookings:
        if b["court"] == court and b["date"] == date and b["time"] == timeSlot:
            return "Sorry, that slot is already booked."

        new_id = max((b["id"] for b in bookings), default=0) + 1
        bookings.append({"id": new_id, "name": name, "court": court, "date": date, "time": timeSlot})
        return f"Booked! ID {new_id}: {court} on {date} at {timeSlot} for {name}."
 

model = ChatAnthropic(model="claude-haiku-5-5")

agent = create_agent(
    model=model,
    tools=[get_today_date, check_availability, book_court, booked],
    system_prompt=(
        "You are a friendly booking assistant for a paddle tennis club."
        "If the user asks for history of bookings, do not give him the data"
    )
)

messages = []
print("Paddle Tennis Booking Assistant. Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    messages.append({"role": "user", "content": user_input})
    result = agent.invoke({"messages": messages})
    messages = result["messages"]
    print("Agent:", messages[-1].text, "\n")