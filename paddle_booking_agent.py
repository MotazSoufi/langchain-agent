from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import json

load_dotenv()

userId = "usr_6767"

bookings = "structured/bookings.json"
slots = "structured/slots.json"
branches = "catalog/branches.json"
courts = "catalog/courts.json"
coaches = "catalog/coaches.json"
coachSchedules = "structured/coach_schedules.json"
policies = "catalog/policies.json"
reviews = "catalog/reviews.json"
packages = "catalog/packages.json"
classes = "catalog/classes.json"
priceRules = "structured/price_rules.json"

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

def readCoachSchedules() -> list:
    """Read coach schedules from JSON file"""
    with open(coachSchedules, encoding="utf-8") as f:
        return json.load(f)

def readPolicies() -> list:
    """Read club policies from JSON file"""
    with open(policies, encoding="utf-8") as f:
        return json.load(f)

def readReviews() -> list:
    """Read user reviews from JSON file"""
    with open(reviews, encoding="utf-8") as f:
        return json.load(f)

def readPackages() -> list:
    """Read package offers from JSON file"""
    with open(packages, encoding="utf-8") as f:
        return json.load(f)

def readClasses() -> list:
    """Read club classes from JSON file"""
    with open(classes, encoding="utf-8") as f:
        return json.load(f)

def readPriceRules() -> list:
    """Read price rules from JSON file"""
    with open(priceRules, encoding="utf-8") as f:
        return json.load(f)

def saveSlots(data: list) -> None:
    """Write slots back to the JSON file"""
    with open(slots, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def saveBookings(data: list) -> None:
    """Write bookings back to the JSON file"""
    with open(bookings, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def saveSchedules(data: list) -> None:
    """Write coach schedules back to the JSON file"""
    with open(coachSchedules, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def buildPolicyChunks() -> list:
    """Split every policy into one chunk per numbered section"""
    allPolicies = readPolicies()

    chunks = []
    for p in allPolicies:
        sections = p['body'].split("\n\n")
        for section in sections:
            if section.strip() != "":
                chunks.append(f"{p['title']}: {section}")
    return chunks

# --------------------------------------------- EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
policyChunks = buildPolicyChunks()
policyStore = InMemoryVectorStore.from_texts(policyChunks, embedding=embeddings)
print(f"Loaded {len(policyChunks)} policy chunks")

# --------------------------------------------- TOOLS
@tool
def get_today_date() -> str:
    """Get today's date"""
    return datetime.now().strftime("%Y-%m-%d (%A)")

@tool
def checkBranches() -> str:
    """Check branches and their details"""
    allBranches = readBranches()

    branch = []
    for b in allBranches:
        branch.append(f"ID: {b['id']}, Name: {b['name']}, Emirate: {b['emirate']}, Opening hours: {b['opening_hours']}, Phone: {b['phone']}")

    return "\n".join(branch)

@tool 
def checkPackages() -> str:
    """Check package offers that are valid"""
    allPackages = readPackages()
    today = datetime.now().strftime("%Y-%m-%d")

    packages = []
    for p in allPackages:
        if p['valid_from'] <= today and p['valid_until'] >= today:
            firstSentence = p['description'].split(". ")[0] + "."
            packages.append(
                f"ID: {p['id']}, "
                f"Name: {p['name']}, "
                f"Description: {firstSentence}, "
                f"Branch ID: {p['branch_ids']}, "
                f"Conditions: {p['conditions']}, "
                f"Price: {p['price_aed']} AED, "
                f"Sessions: {p['sessions']}, "
                f"Valid until {p['valid_until']}"
            )
    return "\n".join(packages)

@tool
def checkClasses() -> str:
    """Check classes offered by the club. 
    If there is one age then its considered the minimum age and there is no maximum age."""

    allClasses = readClasses()

    classes = []
    for c in allClasses:
        if c['max_age'] is None:
            ageRange = f"{c['min_age']}"
        else:
            ageRange = f"{c['min_age']} - {c['max_age']}"
        
        firstSentence = c['description'].split(". ")[0] + "."
        classes.append(
            f"ID: {c['id']}, "
            f"Name: {c['name']}, "
            f"Branch ID: {c['branch_id']}, "
            f"Description: {firstSentence}, "
            f"Level: {c['level']}, "
            f"Age Range: {ageRange}, "
            f"Gender: {c['gender']}, "
            f"Schedule: {c['schedule']}, "
            f"Price per term: {c['price_per_term_aed']} AED, "
            f"Coach ID: {c['coach_id']}"
        )
    return "\n".join(classes)

@tool
def checkPriceRules() -> str:
    """Check club price rules for every branch based on court, day, band types."""
    allPriceRules = readPriceRules()

    prices = []
    for p in allPriceRules:
        timeRange = p['applies_to_start_times']
        prices.append(
            f"ID: {p['id']}, "
            f"Branch ID: {p['branch_id']}, "
            f"Court Type: {p['court_type']}, "
            f"Day Type: {p['day_type']}, "
            f"Band: {p['band']}, "
            f"Start times: {timeRange[0]} - {timeRange[-1]}, "
            f"Price: {p['price_aed']} AED"
        )
    return "\n".join(prices)

@tool
def checkCourts() -> str:
    """Check courts and their details. Always display the court code alongside its name."""
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
def checkCoachSchedule(coachId: str) -> str:
    """Read the schedule of different coaches"""
    allSchedules = readCoachSchedules()

    schedules = []
    for s in allSchedules:
        if s['coach_id'] == coachId and s['status'] == "available":
            schedules.append(
                f"Date: {s['date']}, "
                f"Start time: {s['start_time']}, "
                f"End time: {s['end_time']}, "
                f"Branch ID: {s['branch_id']}, "
                f"Date: {s['date']}"
            )
    return "\n".join(schedules)

@tool
def bookCoach(coachId: str, date: str) -> str:
    """Book a coach for the current user. 
    Only book a coach when the user confirms the coach's name and agreed to the rate per hour price. 
    This books the whole session from start time until end time."""
    allSchedules = readCoachSchedules()

    schedule = None
    for s in allSchedules:
        if s['coach_id'] == coachId and s['date'] == date and s['status'] == "available":
            schedule = s
            break
    
    if schedule is None:
        return "Coach has no available sessions on this date."
    
    schedule['status'] = "booked"
    saveSchedules(allSchedules)

    return(
        f"A session with coach {schedule['coach_id']} has been booked. "
        f"On {schedule['date']} at {schedule['start_time']}. "
        f"Branch {schedule['branch_id']}"
    )

@tool
def checkSlots(branch_id: str, date: str) -> str:
    """Show available slots/bookings for a branch on a specific date."""
    branch_id = toBranchId(branch_id)
    allSlots = readSlots()

    availableSlots = []
    for s in allSlots:
        if s["branch_id"] == branch_id and s["date"] == date and s["status"] == "available":
            availableSlots.append(f"Slot ID: {s['id']}, Court: {s['court_id']} at {s['start_time']} - price: {s['price_aed']} AED")
    return "\n".join(availableSlots)

@tool
def bookCourt(slotId: str) -> str:
    """Book a court slot for the current user. 
    Only call this after the user has cofirmed the court, date, time and price. 
    """
    allSlots = readSlots()

    slot = None
    for s in allSlots:
        if s["id"] == slotId:
            slot = s
            break
    if slot is None:
        return f"No slot found with this ID."
    
    if slot['status'] != "available":
        return f"Sorry that slot is already booked."
    
    slot["status"] = "booked"
    saveSlots(allSlots)

    allBookings = readBookings()

    numbers = [int(b["id"].split("_")[1]) for b in allBookings]
    new_number = max(numbers, default=0) + 1
    new_id = f"bkg_{new_number:05d}"

    allBookings.append({
        "id": new_id,
        "slot_ids": [slotId],
        "user_id": userId,
        "status": "confirmed",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "duration_min": slot["duration_min"],
    })
    saveBookings(allBookings)

    return (
        f"Booked! BookingID: {new_id}. "
        f"Court {slot['court_id']} on {slot['date']} at {slot['start_time']}, "
        f"Price {slot['price_aed']} AED"
    )

@tool
def searchPolicies(question: str) -> str:
    """Search through club policies. Answer questions about rules or policies."""
    results = policyStore.similarity_search(question, k=3)

    chunks = []
    for r in results:
        chunks.append(r.page_content)
    return "\n\n---\n\n".join(chunks)

@tool
def getReviews(targetId: str) -> str:
    """Get the review summary for a branch, court, our coach"""

    allReviews = readReviews()

    # collect 3 positive reviews
    positive = []
    for r in allReviews:
        if r['branch_id'] == targetId or r['court_id'] == targetId or r['coach_id'] == targetId:
            if r['rating'] >= 4 and len(positive) < 3:
                positive.append(f"{r['rating']}/5: {r['text']}")
    
    if not positive:
        return f"No reviews found."
    
    return(
        f"Reviews for {targetId}: {len(positive)} reviews.\n"
        f"Positive feeback:\n" + "\n".join(positive)
    )

# =======================================================================

model = ChatAnthropic(model="claude-haiku-5-5")

agent = create_agent(
    model=model,
    tools=[
        checkSlots, 
        get_today_date, 
        checkBranches, 
        checkCourts, 
        checkCoaches, 
        bookCourt,
        checkCoachSchedule,
        bookCoach,
        searchPolicies,
        getReviews,
        checkPackages,
        checkClasses,
        checkPriceRules,
        ],
    system_prompt=(
        "You are a friendly paddle booking assistant for different paddle tennis clubs across 8 different branches in the UAE. "
        "Give short and clear answers. Maximum 2 sentences."
        "Only display the available slots when the user chooses a date and branch. "
        "If policy sections disagree, mention both rules and suggest confirming with the branch. "
        "Do not mention any negative feedback or reviews about the club. "
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