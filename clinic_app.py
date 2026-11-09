import html
import re
from datetime import datetime
import streamlit as st
import clinic_agent
from clinic_agent import agent

st.set_page_config(page_title="Velora Clinic", page_icon="💬")

st.markdown("""
<style>
.stApp { background-color: #2c363b; }
header[data-testid="stHeader"] { display: none; }
.block-container { max-width: 760px; padding-top: 0; padding-bottom: 6rem; }

.wa-header {
    position: sticky; top: 0; z-index: 10;
    background: #008069; color: white;
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px; margin: 0 -1rem 12px -1rem;
}
.wa-avatar {
    width: 40px; height: 40px; border-radius: 50%;
    background: #dfe5e7; color: #008069;
    display: flex; align-items: center; justify-content: center;
    font-weight: bold; font-size: 18px;
}
.wa-name { font-weight: 600; font-size: 16px; }
.wa-status { font-size: 12px; opacity: 0.85; }

.row { display: flex; margin: 4px 0; }
.row.user { justify-content: flex-end; }
.row.bot  { justify-content: flex-start; }
.bubble {
    max-width: 75%; padding: 6px 9px 8px 9px;
    border-radius: 8px; font-size: 15px; line-height: 1.4;
    box-shadow: 0 1px 0.5px rgba(0,0,0,0.13); color: #111b21;
    text-align: start;
}
.bubble.user { background: #d9fdd3; border-top-right-radius: 0; }
.bubble.bot  { background: #ffffff; border-top-left-radius: 0; }
.time { display: block; text-align: right; font-size: 11px; color: #667781; margin-top: 2px; }


.stButton > button {
    background: #ffffff; color: #008069;
    border: 1px solid #008069; border-radius: 18px;
    font-size: 14px; padding: 4px 12px;
    height: auto; min-height: 38px;        
    white-space: normal;                   
}
.stButton > button p {
    white-space: normal;      
    overflow-wrap: anywhere;                
}
.stButton > button:hover { background: #d9fdd3; color: #008069; border-color: #008069; }

/* typing bubble: 3 bouncing dots */
.typing { display: flex; gap: 4px; padding: 10px 12px; }
.typing span {
    width: 8px; height: 8px; border-radius: 50%;
    background: #8696a0;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0);    opacity: 0.4; }
    30%           { transform: translateY(-5px); opacity: 1;   }
}

[data-testid="stBottom"] > div { background: #2c363b; }
[data-testid="stChatInput"] { border-radius: 24px; }
[data-testid="stChatInput"] textarea { text-align: start; unicode-bidi: plaintext; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------- HELPERS
def toHtml(text: str) -> str:
    """Make agent text safe for HTML and keep simple formatting."""
    text = html.escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?m)^\s*[-*] ", "• ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)    
    return text.replace("\n", "<br>")

def bubble(role: str, text: str, time: str) -> None:
    """Draw one chat bubble. role is 'user' or 'bot'."""
    ticks = " ✓✓" if role == "user" else ""
    if re.search(r"[\u0600-\u06FF]", text):
        direction = "rtl"
    else:
        direction = "ltr"
    st.markdown(
        f'<div class="row {role}"><div class="bubble {role}" dir="{direction}">{toHtml(text)}'
        f'<span class="time">{time}{ticks}</span></div></div>',
        unsafe_allow_html=True,
    )

def startChat(userChoice: str) -> None:
    """Save the chosen user and start a fresh chat."""
    st.session_state.user = userChoice
    st.session_state.history = []
    st.session_state.chat = [("bot", "Hi! I'm the Velora Clinic assistant. How can I help you today?",
                              datetime.now().strftime("%H:%M"))]

# ---------------------------------------------------------- USER POP-UP
users = sorted({b['user_id'] for b in clinic_agent.readBookings()})
users.append("usr_9999")

@st.dialog("Choose a user")
def pickUser():
    choice = st.selectbox("Who do you want to chat as?", users, index=users.index("usr_1001"))
    if st.button("Start chatting"):
        startChat(choice)
        st.rerun()

if "user" not in st.session_state:
    pickUser()
    st.stop()

clinic_agent.userId = st.session_state.user

# ---------------------------------------------------------- HEADER
st.markdown(f"""
<div class="wa-header">
    <div class="wa-avatar">V</div>
    <div><div class="wa-name">Velora Clinic</div>
    <div class="wa-status">online · chatting as {st.session_state.user}</div></div>
</div>
""", unsafe_allow_html=True)

if st.button("Switch user"):
    del st.session_state.user
    st.rerun()

# ---------------------------------------------------------- CHAT
for role, text, time in st.session_state.chat:
    bubble(role, text, time)

userInput = st.chat_input("Type a message")

if userInput:
    now = datetime.now().strftime("%H:%M")
    bubble("user", userInput, now)
    st.session_state.chat.append(("user", userInput, now))

    st.session_state.history.append({"role": "user", "content": userInput})
    
    typing = st.empty()
    typing.markdown(
        '<div class="row bot"><div class="bubble bot typing">'
        '<span></span><span></span><span></span></div></div>',
        unsafe_allow_html=True,
    )
    result = agent.invoke({"messages": st.session_state.history})
    typing.empty()

    st.session_state.history = result["messages"]

    reply = result["messages"][-1].text

    now = datetime.now().strftime("%H:%M")
    bubble("bot", reply, now)
    st.session_state.chat.append(("bot", reply, now))
    st.rerun()                                                                          