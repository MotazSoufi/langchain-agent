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
    "yas": "br_yas", "yas island": "br_yas",
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

def typeWarning(c: dict) -> str:
    """Return a warning if a court's type contradicts its own description, otherwise empty test."""
    text = (c['description'] or "").lower()

    if c['type'] == "indoor":
        for word in ["open-air", "open air", "nothing overhead", "no shade or roof", "open to the sky"]:
            if word in text:
                return " (WARNING: listed as indoor, but its description says otherwise. Suggest confirming with the branch.)"
    
    if c['type'] == "outdoor":
        for word in ["fully enclosed", "air-conditioned", "sealed roof"]:
            if word in text:
                return " (WARNING: listed as outdoor, but its description says otherwise. Suggest confirming with the branch.)"
    
    return ""

def cleanAges(c: dict):
    """Return (minAge, maxAge), or none if the ages in the data can't be trusted."""
    minAge = c['min_age']
    maxAge = c['max_age']

    if minAge is None:
        return None
    if maxAge is not None and minAge > maxAge:
        return None
    if minAge < 5:
        return None
    return (minAge, maxAge)

def coachLanguages(c: dict) -> str:
    """Combine the lagnuages list with any languages mentioned in the bio."""
    allLanguages = ["English", "Arabic", "Spanish", "Italian", "French", "Russian", "Portuguese", "Hindi", "Urdu", "Malayalam", "Japanese"]

    langs = list(c['languages'] or [])
    for lang in allLanguages:
        if lang in c['bio'] and lang not in langs:
            langs.append(lang)
    return ', '.join(langs)

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
    return datetime.now().strftime("%Y-%m-%d (%A) %H:%M")

@tool
def checkBranches(emirate: str = "") -> str:
    """List branches with area, emirate, court count, amenities, opening hours, and phone number."""
    emirate = emirate.strip().lower()

    branch = []
    for b in readBranches():
        if emirate != "" and b['emirate'].lower() != emirate:
            continue
        
        allHours = set(b['opening_hours'].values())
        hours = f"Daily hours: {list(allHours)[0]}"

        branch.append(
            f"ID: {b['id']}, "
            f"Name: {b['name']}, "
            f"Area: {b['area']}, {b['emirate']}, "
            f"Courts: {b['court_count']}, "
            f"Amenities: {', '.join(b['amenities'])}, "
            f"Hours: {hours}, "
            f"Phone: {b['phone']}"
        )
    if not branch:
        return "No branches found with those filters."
    return "\n".join(branch)

@tool
def getBranchDetails(branch_id: str) -> str:
    """Get the full description of one branch. Use when user asks for more details about a specific branch."""
    branch_id = toBranchId(branch_id)
    for b in readBranches():
        if b['id'] == branch_id:
            return f"{b['name']}: {b['description']}"
    return f"No branch found."

@tool 
def checkPackages(branch_id: str = "", name: str = "") -> str:
    """Check packages with price, sessions, branches, and whether they are available today."""
    if branch_id != "":
        branch_id = toBranchId(branch_id)
    name = name.strip().lower()
    today = datetime.now().strftime("%Y-%m-%d")

    packages = []
    for p in readPackages():
        if branch_id != "" and branch_id not in p['branch_ids']:
            continue
        if name != "" and name not in p['name'].lower():
            continue
        if p['valid_from'] <= today <= p['valid_until']:
            packages.append(
                f"ID: {p['id']}, "
                f"Name: {p['name']}, "
                f"Branches: {', '.join(p['branch_ids'])}, "
                f"Price: {p['price_aed']} AED for {p['sessions']} sessions, "
                f"Conditions: {p['conditions']}, "
                f"Valid until: {p['valid_until']}"
            )

        elif today > p['valid_until']:
            packages.append(f"Name: {p['name']} expired on {p['valid_until']}")
        else:
            packages.append(f"Name: {p['name']} is not available yet. Will be available on {p['valid_from']}")

    if not packages:
        return "No packages found with those filters."
    return "\n".join(packages)

@tool
def getPackageDetails(packageId: str) -> str:
    """Get the description of one package. Use when the user asks for more details about a specific package."""
    for p in readPackages():
        if p['id'] == packageId:
            return f"{p['name']}: {p['description']}"
    return f"No package found."
    

@tool
def checkClasses(branch_id: str = "", level: str = "", gender: str = "", age: int = 0) -> str:
    """List classes with level, ages, gender, schedule, price per term and coach.
    For the full description of one class use getClassDetails"""

    if branch_id != "":
        branch_id = toBranchId(branch_id)
    level = level.strip().lower()
    gender = gender.strip().lower()

    classes = []
    coachNames = {}
    for co in readCoaches():
        coachNames[co['id']] = co['name']

    for c in readClasses():
        if branch_id != "" and c['branch_id'] != branch_id:
            continue
        if level != "" and c['level'] != level and c['level'] != "all_levels":
            continue
        if gender != "" and c['gender'] != gender and c['gender'] != "mixed":
            continue

        ages = cleanAges(c)
        if age != 0 and ages is not None:
            minAge, maxAge = ages
            if age < minAge:
                continue
            if maxAge is not None and age > maxAge:
                continue
        
        if ages is None:
            ageRange = "not reliable, check description"
        elif ages[1] is None:
            ageRange = f"{ages[0]}+"
        else:
            ageRange = f"{ages[0]}-{ages[1]}"

        classes.append(
            f"ID: {c['id']}, "
            f"Name: {c['name']}, "
            f"Branch: {c['branch_id']}, "
            f"Level: {c['level']}, "
            f"Ages: {ageRange}, "
            f"Gender: {c['gender']}, "
            f"Schedule: {c['schedule']}, "
            f"Price: {c['price_per_term_aed']} AED/term, "
            f"Coach: {coachNames.get(c['coach_id'], c['coach_id'])}"
        )

    if not classes:
        return "No classes found with those filters."
    return "\n".join(classes)

@tool
def getClassDetails(classId: str) -> str:
    """Get the full description of one class when user asks to describe a specific class."""
    junkWords = ["javascript", "<div", "newsletter", "error code", "unsubscribe"]

    for c in readClasses():
        if c['id'] == classId:
            text = (c['description'] or "").lower()
            for word in junkWords:
                if word in text:
                    return f"{c['name']}: no description available. Suggest contacting the branch."
            return f"{c['name']}: {c['description']}"
    return f"No class found with this ID."

@tool
def checkPriceRules(branch_id: str = "", courtType: str = "", dayType: str = "", band: str = "") -> str:
    """General court prices by branch, court type, day type and time band. 
    For the price of a specific slot use checkSlots."""
    
    if branch_id != "":
        branch_id = toBranchId(branch_id)
    courtType = courtType.strip().lower()
    dayType = dayType.strip().lower()
    band = band.strip().lower()

    prices = []
    for p in readPriceRules():
        if branch_id != "" and p['branch_id'] != branch_id:
            continue
        if courtType != "" and p['court_type'] != courtType:
            continue
        if dayType != "" and p['day_type'] != dayType:
            continue
        if band != "" and p['band'] != band:
            continue

        times = p['applies_to_start_times']
        prices.append(
            f"{p['branch_id']}, {p['court_type']}, {p['day_type']}, "
            f"{p['band']} ({times[0]}-{times[-1]}): {p['price_aed']} AED"
        )

    if not prices:
        return "No price rules found with those filters."
    return "\n".join(prices)

@tool
def checkCourtPrices(branch_id:str, courtType: str = "") -> str:
    """Court prices per hour at one branch by court type, day type, and time band, 
    with the cheapest and most expensive court for each. Use this for any question about court prices, costs, booking fees or rates."""
    branch_id = toBranchId(branch_id)
    courtType = courtType.strip().lower()

    courtInfo = {}
    for c in readCourts():
        courtInfo[c['id']] = (c['type'], c['code'])
    
    bands = {}
    for p in readPriceRules():
        times = p ['applies_to_start_times']
        for t in times:
            bands[t] = f"{p['band']} ({times[0]}-{times[-1]})"

    prices = {}
    for s in readSlots():
        if s['branch_id'] != branch_id:
            continue
        cType, code = courtInfo[s['court_id']]
        if courtType != "" and cType != courtType:
            continue

        weekday = datetime.strptime(s['date'], "%Y-%m-%d").strftime("%a")
        if weekday in ["Fri", "Sat"]:
            dayType = "weekend (Fri-Sat)"
        else:
            dayType = "weekday (Sun-Thu)"
        
        band = bands.get(s['start_time'], "other")
        key = (cType, dayType, band)
        if key not in prices:
            prices[key] = []
        prices[key].append((s['price_aed'], code))

    lines = []
    for (cType, dayType, band), values in prices.items():
        lowest = min(values)
        highest = max(values)
        lines.append(
            f"{cType}, {dayType}, {band}: "
            f"{lowest[0]} AED ({lowest[1]}) to {highest[0]} AED ({highest[1]})"
        )
    
    if not lines:
        return "No prices found for this branch."
    lines.sort()
    return "\n".join(lines)

@tool
def checkCourts(branch_id: str = "", courtType: str = "") -> str:
    """Check courts and their details. Always display the court code alongside its name. 
    For prices use checkCourtPrices. 
    For free times use checkSlots. To describe one court, use getCourtDetails."""
    
    if branch_id != "":
        branch_id = toBranchId(branch_id)
    courtType = courtType.strip().lower()

    court = []
    for c in readCourts():
        if branch_id != "" and c['branch_id'] != branch_id:
            continue
        if courtType != "" and c['type'] != courtType:
            continue

        court.append(
            f"ID: {c['id']}, "
            f"Code: {c['code']}, "
            f"Branch ID: {c['branch_id']}, "
            f"Name: {c['name']}, "
            f"Type: {c['type']}{typeWarning(c)}, "
            f"Surface: {c['surface']}, "
            f"Walls: {c['walls']}, "
            f"Lighting: {c['lighting']}"
        )
    if not court:
        return "No courts found with those filters."
    return "\n".join(court)

@tool
def getCourtDetails(courtId: str) -> str:
    """Describe one court in detail when user asks to describe a specific court."""
    randomText = ["cookie", "404", "javascript", "<div", "newsletter", "skip to main content", "loading availability", "home"]
    for c in readCourts():
        if c['id'] == courtId:
            facts = (
                f"{c['name']} ({c['code']}) at {c['branch_id']}: "
                f"type {c['type']}, surface {c['surface']}, walls {c['walls']}, lighting {c['lighting']}."
            )

            description = c['description'] or ""
            isRandom = False
            for word in randomText:
                if word in description.lower():
                    isRandom = True

            if isRandom or description == "":
                return facts + "No reliable description is available for this court."
            return facts + "Description: " + description
    
    return f"No court found with this ID."


@tool
def checkCoaches(branch_id: str = "") -> str:
    """Check coaches with specialities, languages, level focus, experience and hourly rate. 
    If languages or experience say 'check bio' use getCoachDetails to read the real value."""
    allCoaches = readCoaches()

    if branch_id != "":
        branch_id = toBranchId(branch_id)

    coach = []
    for c in allCoaches:
        if branch_id != "" and c['branch_id'] != branch_id:
            continue
        
        langs = coachLanguages(c)
        
        years = c['years_experience']
        if years is None or years > 50:
            years = "not reliable, check bio"
        else:
            years = f"{years} years"

        coach.append(
            f"ID: {c['id']}, " 
            f"Name: {c['name']}, "
            f"Branch ID: {c['branch_id']}, "
            f"Specialties: {', '.join(c['specialties'])}, "
            f"Languages: {langs}, "
            f"Level Focus: {c['level_focus']}, "
            f"Experience: {years}, "
            f"Rate Per Hour: {c['rate_per_hour_aed']} AED"
        )
    if not coach:
        return f"No coaches found."
    return "\n".join(coach)

@tool
def getCoachDetails(coachId: str) -> str:
    """Get the full bio of one coach, only when the user asks about a specific coach.
     Or confirm a coach's real languages and years of experience."""

    for c in readCoaches():
        if c['id'] == coachId:
            return f"{c['name']}: {c['bio']}"
    return f"No coach found with this ID."

@tool
def checkCoachSchedule(coachId: str) -> str:
    """Read the schedule of different coaches to show upcoming available sessions."""
    allSchedules = readCoachSchedules()
    today = datetime.now().strftime("%Y-%m-%d")    
    nowTime = datetime.now().strftime("%H:%M") 

    schedules = []
    for s in allSchedules:
        if s['coach_id'] == coachId and s['status'] == "available" and s['date'] >= today:
            if s['date'] == today and s['end_time'] <= nowTime:
                continue
            schedules.append(
                f"Schedule ID: {s['id']}, "
                f"Date: {s['date']}, "
                f"Start time: {s['start_time']}, "
                f"End time: {s['end_time']}, "
                f"Branch ID: {s['branch_id']}, "
            )
    if not schedules:
        return f"No upcoming available sessions found."
    return "\n".join(schedules)

@tool
def bookCoach(coachId: str, date: str) -> str:
    """Book a coach for the current user. 
    Only book a coach when the user confirms the coach's name and agreed to the rate per hour price. 
    This books the whole session from start time until end time."""
    
    today = datetime.now().strftime("%Y-%m-%d")
    nowTime = datetime.now().strftime("%H:%M")
    if date < today:
        return f"That date has already passed. Please choose an upcoming date."
    allSchedules = readCoachSchedules()

    schedule = None
    for s in allSchedules:
        if s['coach_id'] == coachId and s['date'] == date and s['status'] == "available":
            schedule = s
            break
    
    if schedule is None:
        return "Coach has no available sessions on this date."
    if schedule['date'] == today and schedule['start_time'] <= nowTime:
        return "Sorry, that session has already started."
    if schedule['status'] != "available":
        return f"Sorry, that is already booked."
    
    schedule['status'] = "booked"
    saveSchedules(allSchedules)

    return(
        f"A session with coach {schedule['coach_id']} has been booked. "
        f"On {schedule['date']} at {schedule['start_time']}. "
        f"Branch {schedule['branch_id']}"
    )

@tool
def checkSlots(branch_id: str, date: str, fromTime: str = "00:00") -> str:
    """Show available slots/bookings for a branch on one date, grouped by court."""
    branch_id = toBranchId(branch_id)

    today = datetime.now().strftime("%Y-%m-%d")
    nowTime = datetime.now().strftime("%H:%M")

    if date < today:
        return f"This date has passed. Bookings can only be made for today or later."

    courtNames = {}
    for c in readCourts():
        courtNames[c['id']] = f"{c['name']} ({c['code']}, {c['type']}{typeWarning(c)})"

    byCourt = {}
    for s in readSlots():
        if s['branch_id'] == branch_id and s['date'] == date and s['status'] == "available":
            if date == today and s['start_time'] <= nowTime:
                continue
            if s['start_time'] < fromTime:
                continue
            courtName = courtNames.get(s['court_id'], s['court_id'])
            if courtName not in byCourt:
                byCourt[courtName] = []
            byCourt[courtName].append(f"{s['start_time']} {s['price_aed']} AED [{s['id']}]")
    
    if not byCourt:
        return f"No available slots at this branch on this date."

    lines = []
    for courtName, times in byCourt.items():
        lines.append(f"{courtName}: " + ", ".join(times))
    return "\n".join(lines)

@tool
def bookCourt(slotIds: list[str]) -> str:
    """Book one or more court slots for the current user as one booking. 
    Only call this after the user has confirmed every court, date, time and price. 
    """
    allSlots = readSlots()
    today = datetime.now().strftime("%Y-%m-%d")
    nowTime = datetime.now().strftime("%H:%M")

    # checking every slot before booking
    chosenSlots = []

    for slotId in slotIds:
        slot = None
        for s in allSlots:
            if s['id'] == slotId:
                slot = s
                break

        if slot is None:
            return f"No slot found with this ID."
        if slot in chosenSlots:
            return f"Slot {slotId} was sent twice"    
        if slot['status'] != "available":
            return f"Sorry that slot is already booked."
        if slot['date']  < today or (slot['date'] == today and slot ['start_time'] <= nowTime):
            return "Sorry, that slot has already passed."

        chosenSlots.append(slot)

    # grouping the slots by court and date
    groups = {}
    for slot in chosenSlots:
        key = (slot['court_id'], slot['date'])
        if key not in groups:
            groups[key] = []
        groups[key].append(slot)

    for slot in chosenSlots:
        slot["status"] = "booked"
    saveSlots(allSlots)

    allBookings = readBookings()
    numbers = [int(b["id"].split("_")[1]) for b in allBookings]
    nextNumber = max(numbers, default=0) + 1

    summary = []
    grandTotal = 0
    for (courtId, date), groupSlots in groups.items():
        newId = f"bkg_{nextNumber:05d}"
        nextNumber = nextNumber + 1

        groupPrice = 0
        times = []
        groupIds = []
        for slot in groupSlots:
            groupPrice = groupPrice + slot['price_aed']
            times.append(slot['start_time'])
            groupIds.append(slot['id'])

        allBookings.append({
            "id": newId,
            "slot_ids": groupIds,
            "user_id": userId,
            "status": "confirmed",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_min": 60 * len(groupSlots),
        })
        grandTotal = grandTotal + groupPrice
        summary.append(f"{newId}: court {courtId} on {date} at {', '.join(times)} ({groupPrice} AED)")

    saveBookings(allBookings)

    return (
        f"Booked {len(groups)} booking(s):\n" + "\n".join(summary) +
        f"\nGrand total: {grandTotal} AED"
    )

@tool
def searchPolicies(question: str) -> str:
    """Search through club policies. Answer questions about rules or policies."""
    results = policyStore.similarity_search(question, k=4)

    chunks = []
    for r in results:
        chunks.append(r.page_content)
    return "\n\n---\n\n".join(chunks)

@tool
def getReviews(targetId: str) -> str:
    """Get the review summary for a branch, court, or coach. 
    targetId must be an exact ID: br_yas, crt_yas_cc04 or cch_diego_fernandez. 
    For a court, call checkCourts first to get its ID."""

    junkWords = ["session has expired", "404", "cookies", "<div", "newsletter", "javascript", "skip to main content"]
    courtCode = ""
    courtBranch = ""

    for c in readCourts():
        if c['id'] == targetId:
            courtCode = c['code']
            courtBranch = c['branch_id']
    
    matched = []
    for r in readReviews():
        if any(word in r['text'].lower() for word in junkWords):
            continue

        if r['branch_id'] == targetId or r['court_id'] == targetId or r['coach_id'] == targetId:
            matched.append(r)
        elif courtCode != "" and r['court_id'] is None and r['branch_id'] == courtBranch and courtCode in r['text'].upper():
            matched.append(r)
    
    if not matched:
        return "No reviews found."
    
    total = len(matched)
    average = sum(r['rating'] for r in matched) / total

    positive = []
    for r in matched:
        if r['rating'] >= 4:
            positive.append(r)
    
    lines = [f"Reviews for {targetId}: average {average:.1f}/5. "]
    lines.append("Top positive:")
    for r in positive[:3]:
        lines.append(f"- {r['rating']}/5: {r['text']}")
    return "\n".join(lines)

@tool
def myBookings(past: bool = False, month: str = "") -> str:
    """Show the current user's court bookings. 
    By default shows upcoming bookings. Set past=True when the user asks about past bookings. 
    For past bookings, month is required in YYYY-MM format (e.g. 2026-10 for October 2026). 
    If the user didn't say which month, call the past=true and no month: you will get the months 
    that have bookings, then ask the user which one they want."""

    slotInfo = {}
    for s in readSlots():
        slotInfo[s['id']] = s
    courtNames = {}
    for c in readCourts():
        courtNames[c['id']] = f"{c['name']} ({c['code']})"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    upcoming = []
    history = []

    for b in readBookings():
        if b['user_id'] != userId:
            continue

        mySlots = []
        for slotId in b['slot_ids']:
            if slotId in slotInfo:
                mySlots.append(slotInfo[slotId])
        
        if not mySlots:
            continue
        mySlots.sort(key=lambda s: (s['date'], s['start_time']))

        first = mySlots[0]
        start = f"{first['date']} {first['start_time']}"
        times = []
        total = 0
        for s in mySlots:
            times.append(s['start_time'])
            total = total + s['price_aed']

        line = (f"{b['id']}: {courtNames.get(first['court_id'], first['court_id'])}, {first['branch_id']}, "
                f"{first['date']} at {', '.join(times)} ({b['duration_min']} min), {total} AED, {b['status']}")

        if b['status'] == "confirmed" and start > now:
            upcoming.append((start, line))
        else:
            history.append((start, line))
        
    if not past:
        if not upcoming:
            return "You have no upcoming bookings."
        upcoming.sort()
        lines = []
        for start, line in upcoming:
            lines.append(line)
        return "Upcoming bookings:\n" + "\n".join(lines)

    if not history:
        return "You have no past bookings."

    if month == "":
        counts = {}
        for start, line in history:
            m = start[:7]
            counts[m] = counts.get(m, 0) + 1
        months = []
        for m in sorted(counts, reverse=True):
            months.append(f"{m} ({counts[m]} booking(s))")
        return ("Ask the user which month they want to see. "
                "Months with past bookings: " + ", ".join(months))

    lines = []
    history.sort(reverse=True)
    for start, line in history:
        if start[:7] == month:
            lines.append(line)

    if not lines:
        return f"You have no past bookings in {month}."
    return f"Your past bookings in {month}:\n" + "\n".join(lines)

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
        getCoachDetails,
        getPackageDetails,
        getBranchDetails,
        getCourtDetails,
        getClassDetails,
        checkCourtPrices,
        myBookings,
        ],
    system_prompt=(
        "You are a friendly paddle booking assistant for different paddle tennis clubs across 8 different branches in the UAE. "
        "Give short and clear answers. "
        "Only display the available slots when the user chooses a date and branch. "
        "If policy sections disagree, mention both rules and suggest confirming with the branch. "
        "Do not mention any negative feedback or reviews about the club. "
        "When booking more than 1 hour, call bookCourt once with all slot ids in the list. Never call it once per slot. "
        "Before calling bookCourt or bookCoach, repeat the exact court or coach, branch, date, time, price, and ask if you should book it. "
        "If two sources disagree such as court's type vs its description, tell the user about both and suggest confirming with the branch. "
        "Only help with baseline padel topics. Politely decline anything else. "
        "Slot and coach schedule data covers 2026-10-05 to 2026-10-19 only. If the user asks for a later date, say no schedule is published yet. "
    )
)

if __name__ == "__main__":
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