import streamlit as st
from paddle_booking_agent import agent

st.set_page_config(page_title="Basline Padel Assistant", page_icon="🎾")
st.title("🎾 Baseline Padel Booking Assistant")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.chat = []

for role, text in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)

userInput = st.chat_input("Ask about courts, coaches, classes or policies...")

if userInput:
    with st.chat_message("user"):
        st.markdown(userInput)
    st.session_state.chat.append(("user", userInput))
    st.session_state.history.append({"role": "user", "content": userInput})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": st.session_state.history})
        st.session_state.history = result["messages"]
        answer = result["messages"][-1].text
        st.markdown(answer)
    st.session_state.chat.append(("assistant", answer))