from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import json
import os
import threading

load_dotenv()

branches = "clinic_data/branches.json"
services = "clinic_data/services.json"
practitioners = "clinic_data/practitioners.json"
slots = "clinic_data/slots.json"
bookings = "clinic_data/bookings.json"
policies = "clinic_data/policies.json"
packages = "clinic_data/packages.json"
reviews = "clinic_data/reviews.json"

userId = "usr_1001"
dataLock = threading.Lock()

BRANCH_ALIASES = {
    "jumeirah": "br_jumeirah", "jumeirah beach road": "br_jumeirah",
    "marina": "br_marina", "dubai marina": "br_marina", "marina walk": "br_marina",
    "abu dhabi": "br_abudhabi", "abudhabi": "br_abudhabi", "reem": "br_abudhabi",
    "al reem": "br_abudhabi", "al reem island": "br_abudhabi",
    "sharjah": "br_sharjah", "majaz": "br_sharjah", "al majaz": "br_sharjah",
    "rak": "br_rak", "ras al khaimah": "br_rak", "al hamra": "br_rak", "hamra": "br_rak",
}

# ----------------------------------------------------- HELPERS

def toBranchId(name: str) -> str:
    """Turn a branch name like 'al reem island' into its branch_id like 'br_abudhabi'"""

    name = name.strip().lower()
    for word in ["velora ladies clinic ", "velora clinic ", "velora "]:
        name = name.replace(word, "")
    if name.startswith("br_"):
        return name
    return BRANCH_ALIASES.get(name, name)

def readBranches() -> list:
    """Read branches from JSON file"""
    with open(branches, encoding="utf-8") as f:
        return json.load(f)

def readServices() -> list:
    """Read services from JSON file"""
    with open(services, encoding="utf-8") as f:
        return json.load(f)    

def readPractitioners() -> list:
    """Read practitioners from JSON file"""
    with open(practitioners, encoding="utf-8") as f:
        return json.load(f)   

def readPolicies() -> list:
    """Read policies from JSON file"""
    with open(policies, encoding="utf-8") as f:
        return json.load(f)   

def readPackages() -> list:
    """Read packages from JSON file"""
    with open(packages, encoding="utf-8") as f:
        return json.load(f)   

def readReviews() -> list:
    """Read reviews from JSON file"""
    with open(reviews, encoding="utf-8") as f:
        return json.load(f)   

def readSlots() -> list:
    """Read slots from JSON file"""
    with open(slots, encoding="utf-8") as f:
        return json.load(f)   

def readBookings() -> list:
    """Read bookings from JSON file"""
    with open(bookings, encoding="utf-8") as f:
        return json.load(f)    

def saveSlots(data: list) -> None:
    """Write slots safely: to a temp file first, then swap it in."""
    tmp = slots + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, slots)

def saveBookings(data: list) -> None:
    """Write bookings safely: to a temp file first, then swap it in."""
    tmp = bookings + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, bookings)

# if session is 40 minutes then 2 slots will be booked
def slotsNeeded(durationMin: int) -> int:
    """How many 30 minute slots needed"""
    return (durationMin + 29) // 30

# this will list the time slots a booking will use
def timesFrom(startTime: str, count: int) -> list:
    """Consecutive slot times, e.g. ('10:00', 3) -> ['10:00', '10:30', '11:00']"""
    t = datetime.strptime(startTime, "%H:%M")
    times = []
    for i in range(count):
        times.append((t + timedelta(minutes=30 * i)).strftime("%H:%M"))
    return times

# obtain the details of a specific service
def findService(serviceId: str):
    """Return one service dict, or None"""
    for s in readServices():
        if s['id'] == serviceId:
            return s
    return None

def findMyBooking(bookingId: str):
    """Return the current user's booking with this ID or none."""
    for b in readBookings():
        if b['id'] == bookingId and b['user_id'] == userId:
            return b
    return None

def lateFee(booking: dict, service: dict) -> int:
    """Policy: free if 24+ hours before. Otherwise 50%, capped at 500 AED for medical treatments."""
    start = datetime.strptime(f"{booking['date']} {booking['start_time']}", "%Y-%m-%d %H:%M")
    hoursLeft = (start - datetime.now()).total_seconds() / 3600
    if hoursLeft >= 24:
        return 0
    fee = booking['price_aed'] * 0.5
    if service['requires_consultation']:
        fee = min(fee, 500)
    return int(fee)

def formatHours(openingHours: dict) -> str:
    """Group days with the same hours, example 'Sun-Thu 09:00-21:00, Fri closed'"""

    days = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]
    groups = []
    for day in days:
        hours = openingHours[day]
        if groups and groups[-1][2] == hours:
            groups[-1][1] = day
        else:
            groups.append([day, day, hours])
    
    parts = []
    for first, last, hours in groups:
        if first == last:
            dayText = first.capitalize()
        else:
            dayText = f"{first.capitalize()}-{last.capitalize()}"
        parts.append(f"{dayText} {hours}")
    return ", ".join(parts)

def buildPolicyChunks() -> list:
    """Split every policy into one chunk per numbered section"""
    chunks = []
    for p in readPolicies():
        sections = p['body'].split("\n\n")
        for section in sections:
            if section.strip() != "":
                chunks.append(f"{p['title']}: {section}")
    return chunks

def toMinutes(hhmm: str) -> int:
    """Turn 'HH:MM' into minutes since midnight, e.g. '13:30' -> 810"""
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)

# ---------------------------------------------------- EMBEDDINGS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
policyChunks = buildPolicyChunks()
policyStore = InMemoryVectorStore.from_texts(policyChunks, embedding=embeddings)
print(f"Loaded {len(policyChunks)} policy chunks")

# ----------------------------------------------------- TOOLS

@tool
def get_today_date() -> str:
    """Get today's date"""
    return datetime.now().strftime("%Y-%m-%d (%A) %H:%M")

@tool
def checkBranches(emirate: str = "", branch_id: str = "") -> str:
    """List the clinic's branches. For a general question, call with no branch_id: 
    you get only names and locations, so give a short overview in one or two sentences, and ask 
    which branch they want to know about. For one branch's opening hours, amenities and 
    phone number, call with its branch_id."""
    emirate = emirate.strip().lower()
    if branch_id != "":
        branch_id = toBranchId(branch_id)

    lines = []
    for b in readBranches():
        if emirate != "" and b['emirate'].lower() != emirate:
            continue
        if branch_id != "" and b['id'] != branch_id:
            continue

        if b['ladies_only']:
            clients = "ladies only"
        else:
            clients = "men and women"

        if branch_id == "":
            lines.append(f"{b['name']} [{b['id']}]: {b['area']}, {b['emirate']}, {clients}")
        else:
            lines.append(
                f"{b['name']}: {b['area']}, {b['emirate']}, {clients}. "
                f"Hours: {formatHours(b['opening_hours'])}. "
                f"Amenities: {', '.join(b['amenities'])}. "
                f"Phone: {b['phone']}"
            )

    if not lines:
        return "No branches found."
    return "\n".join(lines)

@tool
def getBranchDetails(branch_id: str) -> str:
    """Get the full description of one branch. Use when the user asks for more details about a specific branch."""
    branch_id = toBranchId(branch_id)
    for b in readBranches():
        if b['id'] == branch_id:
            return f"{b['name']}: {b['description']}"
    return "No branch found."

@tool
def checkServices(branch_id: str = "", category: str = "") -> str:
    """List services with duration, price, whether a consultation is needed and minimum age. 
    If the user asks generally what services there are, call without a category: you get a short 
    summary of the categories, then ask the user what they are looking for. Never list every service. 
    For the full description, safety info and session gaps use getServiceDetails."""

    if branch_id != "":
        branch_id = toBranchId(branch_id)
    category = category.strip().lower()

    if category == "":
        groups = {}
        for s in readServices():
            if branch_id != "" and branch_id not in s['branch_ids']:
                continue
            if s['category'] not in groups:
                groups[s['category']] = []
            groups[s['category']].append(s['price_aed'])

        if not groups:
            return "No services found at this branch."

        lines = []
        for cat, prices in groups.items():
            lines.append(f"{cat}: {len(prices)} services, {min(prices)}-{max(prices)} AED")
        return ("Ask the user what kind of treatment they are looking for, Categories:\n"
                + "\n".join(lines))

    result = []
    for s in readServices():
        if branch_id != "" and branch_id not in s['branch_ids']:
            continue
        if category != "" and s['category'] != category:
            continue

        if s['requires_consultation']:
            consult = "consultation required"
        else:
            consult = "no consultation"

        gender = ""
        if s['gender'] != "all":
            gender = f", {s['gender']} only"
        
        result.append(
            f"ID: {s['id']}, {s['name']} ({s['category']}): "
            f"{s['duration_min']} min, {s['price_aed']} AED, "
            f"by {s['practitioner_role']}, {consult}, age {s['min_age']}+{gender}"
        )
    if not result:
        return "No services found with those filters."
    return "\n".join(result)

@tool
def getServiceDetails(serviceId: str) -> str:
    """Get the full description of one service: downtime, recommended sessions, days between sessions, 
    when it is not suitable, and which branches offer it."""
    for s in readServices():
        if s['id'] == serviceId:
            contra = ", ".join(s['contraindications']) or "none listed"
            return (
                f"{s['name']}: {s['description']} "
                f"Downtime: {s['downtime']}. "
                f"Recommended sessions: {s['recommended_sessions']}, at least {s['min_days_between_sessions']} days apart. "
                f"Not suitable with: {contra}. "
                f"Available at: {', '.join(s['branch_ids'])}." 
            )
    return "No service found with this ID."

@tool
def checkPractitioners(branch_id:str = "", serviceId: str = "", role: str = "", gender: str = "", language: str = "") -> str:
    """List practitioners. If the user asks generally who works at a branch, call with only branch_id: 
    you get names by role, then ask what treatment they need. 
    With serviceId, role, gender or language you get full details: title, gender, language, 
    working days, experience and services.
    For a practitioner's bio use getPractitionersDetails."""

    if branch_id != "":
        branch_id = toBranchId(branch_id)
    role = role.strip().lower()
    gender = gender.strip().lower()
    language = language.strip().lower()

    if serviceId == "" and role == "" and gender == "" and language == "":
        byRole = {}
        for p in readPractitioners():
            if branch_id != "" and p['branch_id'] != branch_id:
                continue
            if p['role'] not in byRole:
                byRole[p['role']] = []
            byRole[p['role']].append(p['name'])

        if not byRole:
            return "No practitioners found."
        
        lines = []
        for r, names in byRole.items():
            lines.append(f"{r}s: {', '.join(names)}")
        return ("Briefly say whos there by role, in one sentence, without details. "
                "Then ask if the user is looking for a specific treatment. Practitioners:\n" + "\n".join(lines))

    # this makes the service id have the service names next to each one
    serviceNames = {}
    for s in readServices():
        serviceNames[s['id']] = s['name']

    result = []
    for p in readPractitioners():
        if branch_id != "" and p['branch_id'] != branch_id:
            continue
        if serviceId != "" and serviceId not in p['service_ids']:
            continue
        if role != "" and p['role'] != role:
            continue
        if gender != "" and p['gender'] != gender:
            continue

        langs = []
        for l in p['languages']:
            langs.append(l.lower())
        if language != "" and language not in langs:
            continue

        names = []
        for i in p['service_ids']:
            names.append(serviceNames.get(i, i))

        days = []
        for d in p['working_days']:
            days.append(d.capitalize())

        result.append(
            f"ID: {p['id']}, {p['name']} ({p['title']}), "
            f"Branch: {p['branch_id']}, Gender: {p['gender']}, "
            f"Languages: {', '.join(p['languages'])}, "
            f"Works: {', '.join(days)}, "
            f"Experience: {p['years_experience']} years, "
            f"Services: {', '.join(names)}"
        )
    if not result:
        return "No practitioners found with those filters."
    return "\n".join(result)

@tool
def getPractitionerDetails(practitionerId: str) -> str:
    """Get the bio and specialties of one practitioner."""
    for p in readPractitioners():
        if p['id'] == practitionerId:
            return f"{p['name']} ({p['title']}): {p['bio']} Specialties: {', '.join(p['specialties'])}."
    return "No practitioner found with this ID."

@tool
def checkAvailability(serviceId: str, branch_id: str, date: str, preferredTime: str = "", practitionerId: str = "") -> str:
    """Find when a service can be booked at a branch on one date (YYYY-MM-DD). 
    If the user hasn't said what time they want, call without preferredTime: you get a short summary, 
    then ask the user roughly what time suits them. Then call again with preferredTime (HH:MM) 
    to get the closest available start times. Never list every time slot."""
    branch_id = toBranchId(branch_id)
    service = findService(serviceId)

    if service is None:
        return "No service found with this ID. Use checkServices to find it."
    if branch_id not in service['branch_ids']:
        return f"{service['name']} is not offered at this branch. Available at: {', '.join(service['branch_ids'])}."

    today = datetime.now().strftime("%Y-%m-%d")
    nowTime = datetime.now().strftime("%H:%M")
    if date < today:
        return "This date has passed."

    count = slotsNeeded(service['duration_min'])

    free = {}
    for s in readSlots():
        if s['branch_id'] == branch_id and s['date'] == date and s['status'] == "available":
            free[(s['practitioner_id'], s['start_time'])] = True

    options = []
    for p in readPractitioners():
        if p['branch_id'] != branch_id or serviceId not in p['service_ids']:
            continue
        if practitionerId != "" and p['id'] != practitionerId:
            continue
        for (pid, time) in free:
            if pid != p['id']:
                continue
            if date == today and time <= nowTime:
                continue
            allFree = True
            for t in timesFrom(time, count):
                if (pid, t) not in free:
                    allFree = False
            
            if allFree:
                options.append((time, p['name'], p['id']))

    if not options:
        return f"No availability for {service['name']} at this branch on {date}."
    
    header = f"{service['name']} ({service['duration_min']} min, {service['price_aed']} AED) on {date}: "

    if preferredTime == "":
        earliest = min(options)[0]
        latest = max(options)[0]
        return header + (f"{len(options)} start times available between {earliest} and {latest}. "
                         f"Ask the user what time they prefer, then call again with preferredTime.")

    target = toMinutes(preferredTime)
    options.sort(key=lambda o: (abs(toMinutes(o[0]) - target), o[0]))
    lines = []

    for time, name, pid in options[:3]:
        lines.append(f"{time} with {name} [{pid}]")
    return header + f"closest to {preferredTime}:\n" + "\n".join(lines)

@tool
def bookAppointment(serviceId: str, practitionerId: str, date: str, startTime: str) -> str:
    """Book a service for the current user. Only call this after the user confirmed the 
    service, practitioner, branch, date, time, and price. Use checkAvailability first."""

    with dataLock:
        service = findService(serviceId)
        if service is None:
            return "No service found with this ID."

        # check if practitioner exists
        practitioner = None
        for p in readPractitioners():
            if p['id'] == practitionerId:
                practitioner = p
        if practitioner is None:
            return "No practitioner found with this ID."

        # check if mentioned practitioner offers the mentioned service
        if serviceId not in practitioner['service_ids']:
            return f"{practitioner['name']} does not perform {service['name']}."
        
        # check if time passed
        if f"{date} {startTime}" <= datetime.now().strftime("%Y-%m-%d %H:%M"):
            return "That time has already passed."

        allBookings = readBookings()
        hasConsultation = False
        newDate = datetime.strptime(date, "%Y-%m-%d")

        for b in allBookings:
            if b['user_id'] != userId or b['status'] not in ["confirmed", "completed"]:
                continue
            bDate = datetime.strptime(b['date'], "%Y-%m-%d")

            # check if the same treatment is being booked again too soon
            if b['service_id'] == serviceId:
                gap = abs((newDate - bDate).days)
                if gap < service['min_days_between_sessions']:
                    nextDate = bDate + timedelta(days=service['min_days_between_sessions'])
                    return (f"Too soon: you had {service['name']} on {b['date']}. It can be repeated "
                            f"after {service['min_days_between_sessions']} days, from {nextDate.strftime("%Y-%m-%d")}.")

            if b['service_id'] == "svc_skin_analysis" and bDate <= newDate and (newDate - bDate).days <= 180:
                hasConsultation = True
            
        if service['requires_consultation'] and not hasConsultation:
            return f"{service['name']} requires a Skin Analysis Consultation (free) first. Please book that first."

        allSlots = readSlots()
        chosen = []
        for t in timesFrom(startTime, slotsNeeded(service['duration_min'])):
            slot = None
            for s in allSlots:
                if s['practitioner_id'] == practitionerId and s['date'] == date and s['start_time'] == t:
                    slot = s
                    break
            if slot is None or slot['status'] != "available":
                return f"{practitioner['name']} is not free for the full {service['duration_min']} minutes from {startTime}."
            chosen.append(slot)

        for slot in chosen:
            slot['status'] = "booked"
        saveSlots(allSlots)

        numbers = [int(b['id'].split("_")[1]) for b in allBookings]
        newId = f"bkg_{max(numbers, default=0) + 1:05d}"

        endTime = (datetime.strptime(startTime, "%H:%M") + timedelta(minutes=service['duration_min'])).strftime("%H:%M")

        allBookings.append({
            "id": newId,
            "user_id": userId,
            "service_id": serviceId,
            "practitioner_id": practitionerId,
            "branch_id": practitioner['branch_id'],
            "date": date,
            "start_time": startTime,
            "end_time": endTime,
            "slot_ids": [s['id'] for s in chosen],
            "price_aed": service['price_aed'],
            "status": "confirmed",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        saveBookings(allBookings)

        return (f"Booked {newId}: {service['name']} with {practitioner['name']} at {practitioner['branch_id']} "
                f"on {date}, {startTime}-{endTime}. Price: {service['price_aed']} AED.")

@tool
def myBookings(past: bool = False, month: str = "") -> str:
    """Show the current user's appointments. 
    By default shows upcoming appointments. Set past=true when the user asks about the past, completed 
    or cancelled appointments. For past appointments, month is required in YYYY-MM format. 
    If the user didn't say which month, call with past=true and 
    no month: you will get the months that have appointments, then ask teh user which one they want."""

    serviceNames = {}
    for s in readServices():
        serviceNames[s['id']] = s['name']

    practitionerNames = {}
    for p in readPractitioners():
        practitionerNames[p['id']] = p['name']

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    upcoming = []
    history = []
    for b in readBookings():
        if b['user_id'] != userId:
            continue

        start = f"{b['date']} {b['start_time']}"
        status = b['status']
        if status == "confirmed" and start <= now:
            status = "completed"

        line = (f"{b['id']}: {serviceNames.get(b['service_id'], b['service_id'])} "
                f"with {practitionerNames.get(b['practitioner_id'], b['practitioner_id'])}, "
                f"{b['branch_id']}, {b['date']} {b['start_time']}-{b['end_time']}, "
                f"{b['price_aed']} AED, {status}")

        if b['status'] == "confirmed" and start > now:
            upcoming.append((start, line))
        else:
            history.append((start, line))

    if not past:
        if not upcoming:
            return "You have no upcoming appointments."
        upcoming.sort()
        lines = []
        for start, line in upcoming:
            lines.append(line)
        return "Upcoming appointments:\n" + "\n".join(lines)

    if not history:
        return "You have no past appointments."

    if month == "":
        counts = {}
        for start, line in history:
            m = start[:7]
            counts[m] = counts.get(m, 0) + 1
        months = []
        for m in sorted(counts, reverse=True):
            months.append(f"{m} ({counts[m]} appointments(s))")

        return ("Ask the user which month they want to see. "
                "Months with past appointments: " + ", ".join(months))

    history.sort(reverse=True)
    lines = []

    for start, line in history:
        if start[:7] == month:
            lines.append(line)
    
    if not lines:
        return f"You have no past appointments in {month}."
    return f"Your past appointments in {month}:\n" + "\n".join(lines)

@tool
def searchPolicies(question: str) -> str:
    """Search the clinic's policies to answer a SPECIFIC question, e.g. 'can i cancel for free?' 
    or 'how late can i arrive?'. Pass the user's question in plain words. 
    For general questions like 'what are your policies?', use listPolicyTopics tool instead."""

    results = policyStore.similarity_search(question, k=4)

    chunks = []
    for r in results:
        chunks.append(r.page_content)
    return "\n\n---\n\n".join(chunks)

@tool
def listPolicyTopics() -> str:
    """List the titles of the clinic's policies. Use this when the users asks generally about the 
    clinic's policies or rules, then ask which topic they want to know about. 
    For a specific policy questions, use searchPolicies instead."""

    titles = []
    for p in readPolicies():
        titles.append(p['title'])
        return ("Mention a few of these topics in a sentence and ask which one the user wants to know about. "
                "Policy topics: " + ", ".join(titles))

@tool
def checkPackages(branch_id: str = "", packageType: str = "", serviceId: str = "") -> str:
    """List packages, bundles and memberships with what is included, price, savings and branches. 
    Also says if an offer has expired or is not on sale yet. 
    If the user asks generally about packages, don't mention prices and call with no packageType and no serviceId: 
    short overview without prices, then ask the user what they are interested in. 
    For conditions and membership perks use getPackageDetails."""

    if branch_id != "":
        branch_id = toBranchId(branch_id)
    packageType = packageType.strip().lower()
    today = datetime.now().strftime("%Y-%m-%d")

    if packageType == "" and serviceId == "":
        overview = {}
        for p in readPackages():
            if branch_id != "" and branch_id not in p['branch_ids']:
                continue
            if not (p['valid_from'] <= today <= p['valid_until']):
                continue
            if p['type'] not in overview:
                overview[p['type']] = []
            overview[p['type']].append(p['name'])

        if not overview:
            return "No offers are on sale right now."

        lines = []
        for offerType, names in overview.items():
            lines.append(f"{offerType}: {', '.join(names)}")
        return ("Give the user a short overview of the kinds of offers, without prices, "
                "and ask what treatment they are interested in. Offers on sale:\n" + "\n".join(lines))

    serviceNames = {}
    for s in readServices():
        serviceNames[s['id']] = s['name']
    
    result = []
    for p in readPackages():
        if branch_id != "" and branch_id not in p['branch_ids']:
            continue
        if packageType != "" and p['type'] != packageType:
            continue

        itemIds = []
        for item in p['items']:
            itemIds.append(item['service_id'])
        if serviceId != "" and serviceId not in itemIds:
            continue

        if today < p['valid_from']:
            result.append(f"{p['name']}: not on sale yet, available from {p['valid_from']}")
            continue
        if today > p['valid_until']:
            result.append(f"{p['name']}: expired on {p['valid_until']}")
            continue
        
        included = []
        for item in p['items']:
            included.append(f"{item['sessions']}x {serviceNames.get(item['service_id'], item['service_id'])}")

        if p['billing'] == "monthly":
            price = f"{p['price_aed']} AED/month (min {p['min_months']} months)"
        else:
            price = f"{p['price_aed']} AED (save {p['savings_aed']} AED), use within {p['use_within_days']} days"

        result.append(
            f"ID: {p['id']} {p['name']} ({p['type']}): "
            f"{', '.join(included)}, {price}, "
            f"branches: {', '.join(p['branch_ids'])}"
        )
    if not result:
        return "No packages found with those filters."
    return "\n".join(result)

@tool
def getPackageDetails(packageId: str) -> str:
    """Get the description and conditions of one package, bundle or membership, including membership perks."""

    for p in readPackages():
        if p['id'] == packageId:
            text = f"{p['name']}: {p['description']} Conditions: {p['conditions']}"
            if p['type'] == "membership":
                text = text + f" Perks: {', '.join(p['perks'])}."
            return text
    return "No package found with this ID."

@tool
def getReviews(targetId: str) -> str:
    """Get a review summary for a branch, practitioner or service."""

    targetId = targetId.strip().lower()

    matched = []
    for r in readReviews():
        if r['branch_id'] == targetId or r['practitioner_id'] == targetId or r['service_id'] == targetId:
            matched.append(r)
    
    if not matched:
        return (f"No reviews found for '{targetId}'. Use an exact ID like br_marina or prc_sara_haddad.")
    
    total = len(matched)
    average = sum(r['rating'] for r in matched) / total

    positive = []
    negative = []
    for r in matched:
        if r['rating'] >= 4:
            positive.append(r)
        elif r['rating'] <= 2:
            negative.append(r)
    
    neutral = total - len(positive) - len(negative)
    positive.sort(key=lambda r: r['rating'], reverse=True)

    lines = [f"Reviews for {targetId}: {total} reviews, average {average:.1f}/5 "
             f"({len(positive)} positive, {neutral} neutral, {len(negative)} negative)."]
    lines.append("Top positive:")
    for r in positive[:4]:
        lines.append(f"- {r['rating']}/5: {r['text']}")
    if negative:
        lines.append("Top negative:")
        for r in negative[:3]:
            lines.append(f"- {r['rating']}/5: {r['text']}")
    return "\n".join(lines)

@tool
def cancelBooking(bookingId: str, confirm: bool = False) -> str:
    """Cancel one of the current user's upcoming bookings. 
    First call with confirm=False to get the cancellation fee, tell the user, can only call again 
    with confirm=True after the user clearly agrees."""

    with dataLock:
        booking = findMyBooking(bookingId)

        if booking is None:
            return "No booking found with this ID."
        if booking['status'] != "confirmed":
            return f"This booking is already {booking['status']}."
        if f"{booking['date']} {booking['start_time']}" <= datetime.now().strftime("%Y-%m-%d %H:%M"):
            return "This appointment has already started or passed, so it can't be cancelled."

        service = findService(booking['service_id'])
        fee = lateFee(booking, service)

        if not confirm:
            if fee == 0:
                return (f"Cancelling {bookingId} ({service['name']} on {booking['date']} at {booking['start_time']}) "
                        f"is free. Ask the user to confirm.")
            return (f"Cancelling {bookingId} ({service['name']} on {booking['date']} at {booking['start_time']}) "
                    f"is less than 24 hours before the appointment, so a late cancellation fee of {fee} AED applies. "
                    f"Ask the user to confirm.")

        allSlots = readSlots()
        for s in allSlots:
            if s['id'] in booking['slot_ids']:
                s['status'] = "available"
        saveSlots(allSlots)

        allBookings = readBookings()
        for b in allBookings:
            if b['id'] == bookingId:
                b['status'] = "cancelled"
                b['cancellation_fee_aed'] = fee
                b['cancelled_at'] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        saveBookings(allBookings)
        return f"Cancelled {bookingId}. Cancellation fee: {fee} AED."

# ----------------------------------------------------- MODEL

model = ChatAnthropic(model="claude-haiku-5-5")

agent = create_agent(
    model=model,
    tools=[
        get_today_date,
        checkBranches,
        getBranchDetails,
        checkServices,
        getServiceDetails,
        checkPractitioners,
        getPractitionerDetails,
        checkAvailability,
        bookAppointment,
        myBookings,
        searchPolicies,
        listPolicyTopics,
        checkPackages,
        getPackageDetails,
        getReviews,
        cancelBooking,
    ],
    system_prompt=(
        "You are a friendly booking assistant for Velora beauty clinic that has 5 branches across the UAE. "
        "Talk like a warm, caring receptionist. Natural, polite, and human, never robotic. "
        "Keep replies short and conversational. "
        "Always reply in the same language as the user (English or Arabic). "
        "Before booking, make sure you know the service, branch, and date. "
        "Always use checkAvailability tool before bookAppointment tool. "
        "If the user asks who you are or what you can do, answer in one short sentence. "
        "When the user just says hello or makes small talk, reply with a short , friendly greeting "
        "and ask how you can help. Do not list services or options. "
        "Never use bullet points, numbered lists or tables except when showing the user's own bookings from myBookings. "
        "Instead of listing many items, mention at most 2 or 3 options in a sentence, or ask a question to narrow it down. "
        "Never read out IDs (like svc_ or prc_ codes); use names instead. "
        "If the user is upset, unhappy with a treatment, or mentions pain, discomfort or a reaction, "
        "start by showing real care: say you're sorry they are going through this, in your own words, "
        "and keep the reply short and focused on their safety. "
        "Then help: share the clinic's aftercare guidance from searchPolicies if relevant. "
        "If they descrive severe pain, swelling, changes in color or signs of infection, "
        "tell them to contact the clinic immedietly or go to the nearest emergency department. "
        "Never give medical advice, instead recommend a free skin analysis consultation with a doctor "
        "when asked about side effects, pregnancy, medications, health conditions, or whether a treatment is safe. "
        "Only use information from your tools. If a tool does not give the answer, say you don't know "
        "and suggest calling the branch. "
        "Only show the current user's own bookings. "
        "Only help with Velora Clinic topics. Politely decline anything else. "
        "Before calling bookAppointment tool, repeat the service, practitioner, branch, date, time and price. "
        "When replying in Arabic, write service, branch and practitioner names in Arabic, "
        "followed by the English name in brackets. "
        "For services with a minimum age, ask the user's age if you don't know it. "
        "The Sharjah branch is ladies only. If the user wants to book there, confirm they are female. "    )
)

if __name__ == "__main__":
    messages = []
    print("Velora Beauty Clinic Assistant. Type 'exit' to quit.\n")
    
    while True:

        userInput = input("You: ")
        if userInput.lower() == "exit":
            break

        messages.append({"role":"user", "content": userInput})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]
        print("Agent:", messages[-1].text, "\n")

