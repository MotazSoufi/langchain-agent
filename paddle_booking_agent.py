from datetime import datetime
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
import json

load_dotenv()

bookings = "structured/bookings.json"
slots = "structured/slots.json"
branches = "catalog/branches.json"
courts = "catalog/courts.json"
coaches = "catalog/coaches.json"

BRANCH_ALIANSES = {
    "yas": "br_yas", "yas island": "br_yas","Baseline Yas Island": "br_yas",
    "khalifa": "br_khalifa", "khalifa city": "br_khalifa",
    "al ain": "br_alain", "alain": "br_alain",
    "al quoz": "br_alquoz", "alquoz": "br_alquoz",
    "jvc": "br_jvc", "jumeirah village circle": "br_jvc", "jumeirah": "br_jvc",
    "majaz": "br_majaz", "al majaz": "br_majaz",
    "ajman": "br_ajman", "ajman corniche": "br_ajman",
    "rak": "br_rak", "ras al khaimah": "br_rak",
}

# --------------------------------------------- HELPERS
def toBranchId(name: str) -> str:
    """Turn branch name like 'yas island' into its branch_id like 'br_yas'"""
    name = name.strip().lower()
    if name.startswith("br_"):
        return name
    return BRANCH_ALIANSES.get(name, name)

def readBookings() -> list:
    """Read bookings from JSON file"""
    with open(bookings, encoding="utf-8") as f:
        return json.load(f)

def readSlots() -> list:
    """Read slots from JSON file"""
    with open(slots, encoding="utf-8") as f:
        return json.load(f)

def readBranches() -> list:
    """Read branches from JSON file"""
    with open(branches, encoding="utf-8") as f:
        return json.load(f)

def readCourts() -> list:
    """Read courts from JSON file"""
    with open(courts, encoding="utf-8") as f:
        return json.load(f)

def readCoaches() -> list:
    """Read coaches from JSON file"""
    with open(coaches, encoding="utf-8") as f:
        return json.load(f)

# --------------------------------------------- TOOLS
@tool
def get_today_date() -> str:
    """Get today's date"""
    return datetime.now().strftime("%Y-%m-%d")

@tool
def checkBranches() -> str:
    """Check branches and their details"""
    allBranches = readBranches()

    branch = []
    for b in allBranches:
        branch.append(f"ID: {b['id']}, Name: {b['name']}, Emirate: {b['emirate']}, Opening hours: {b['opening_hours']}, Phone: {b['phone']}")

    return "\n".join(branch)

@tool
def checkCourts() -> str:
    """Check courts and their details"""
    allCourts = readCourts()

    court = []
    for c in allCourts:
        court.append(f"ID: {c['id']}, Code: {c['code']}, Branch ID: {c['branch_id']}, Name: {c['name']}, Type: {c['type']}, Price: {c['price_per_hour_aed']}, Description: {c['description']}")
    return "\n".join(court)

@tool
def checkCoaches() -> str:
    """Check coaches and their details"""
    allCoaches = readCoaches()

    coach = []
    for c in allCoaches:
        coach.append(
            f"ID: {c['id']}, " 
            f"Name: {c['name']}, "
            f"Branch ID: {c['branch_id']}, "
            f"Bio: {c['bio']}, "
            f"Specialties: {c['specialties']}, "
            f"Languages: {c['languages']}, "
            f"Level Focus: {c['level_focus']}, "
            f"Experience: {c['years_experience']} years, "
            f"Rate Per Hour: {c['rate_per_hour_aed']} AED, "
        )
    return "\n".join(coach)

@tool
def checkSlots(branch_id: str, date: str) -> str:
    """Show available slots/bookings"""
    branch_id = toBranchId(branch_id)
    allSlots = readSlots()

    availableSlots = []
    for s in allSlots:
        if s["branch_id"] == branch_id and s["date"] == date and s["status"] == "available":
            availableSlots.append(f"Court: {s['court_id']} at {s['start_time']} - price: {s['price_aed']} AED")
    return "\n".join(availableSlots)

model = ChatAnthropic(model="claude-haiku-5-5")

agent = create_agent(
    model=model,
    tools=[checkSlots, get_today_date, checkBranches, checkCourts, checkCoaches],
    system_prompt=(
        "You are a friendly paddle booking assistant for different paddle tennis clubs across 8 different branches in the UAE. "
        "Give short and brief answers. Maximum 2 sentences."
        "Only display the available slots when the user chooses a date and branch. "
    )
)

messages = []
print("Paddle Tennis Booking Assistant. Type 'exit' to quit.\n")

while True:

    userInput = input("You: ")
    if userInput.lower() == "exit":
        break

    messages.append({"role": "user", "content": userInput})
    result = agent.invoke({"messages": messages})
    messages = result["messages"]
    print("Agent:", messages[-1].text, "\n")