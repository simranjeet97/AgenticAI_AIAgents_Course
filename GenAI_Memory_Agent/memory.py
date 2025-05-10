import os
import streamlit as st
from datetime import datetime, timedelta
from mem0 import MemoryClient
from autogen import ConversableAgent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

# Set environment variables
os.environ["MEM0_API_KEY"] = "m0-OvNHLIrlaTlIJlIhHLMVkzLCiSHzXGcIlVYAa95d"
GOOGLE_API_KEY = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# LLM Configuration
llm_config = {
    "model": "gemini-2.0-flash",
    "api_key": GOOGLE_API_KEY,
    "api_type": "google"
}

user_id = "digital_chatbot"

# Initialize MemoryClient once (outside of user input loop)
memory = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

customer_bot = ConversableAgent(
    name="customer_bot",
    system_message="You are a customer service bot who gathers information on query customers are asking. Keep answers clear and concise but explained. If do not know the answer, just return i do not have access,i'm sorry, i cannot, as an ai language model, i don't know, i cannot fulfill this request, i am not sure, i cannot access your personal information",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

web_search_agent = Agent(
    name="WebSearchAgent",
    instructions=[
        "Use DuckDuckGo to search the web for up-to-date and accurate information.",
        "Summarize the most relevant and trustworthy info from the search results.",
    ],
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    add_datetime_to_instructions=True,
)

# --- Streamlit UI Styling ---
st.set_page_config(page_title="Smart Support Chatbot", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #0e0e1a;
            color: #f0f0f0;
        }

        .user-msg, .bot-msg {
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            max-width: 80%;
            font-size: 1.05rem;
            box-shadow: 0 0 12px rgba(0,0,0,0.3);
            animation: fadeIn 0.3s ease-in-out;
        }

        .user-msg {
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: white;
            margin-left: auto;
            text-align: right;
            box-shadow: 0 0 10px #00f2fe88;
        }

        .bot-msg {
            background: linear-gradient(135deg, #232526, #414345);
            color: #f0f0f0;
            margin-right: auto;
            border-left: 4px solid #00f2fe;
        }

        .chat-container {
            padding: 2rem 3rem;
            background-color: #10101a;
            border-radius: 20px;
            margin-top: 2rem;
        }

        .chat-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #00f2fe;
            margin-bottom: 1.5rem;
            text-shadow: 0 0 10px #00f2fe88;
        }

        input[type="text"] {
            background-color: #1e1e2f !important;
            color: white !important;
            border: none !important;
            border-radius: 10px;
            padding: 0.75rem !important;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="chat-title">🤖 Digital Chatbot with Memory + Web Search</div>', unsafe_allow_html=True)

user_input = st.text_input("Ask a question:", placeholder="E.g. List latest Marvel movies", key="input")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Check memory freshness
    relevant_memories = memory.search(user_input, user_id=user_id)
    memory_context = ""
    use_memory = False
    fresh_memories = []

    for m in relevant_memories:
        mem_time_str = m.get("timestamp") or m.get("created_at")
        if mem_time_str:
            try:
                mem_time = datetime.fromisoformat(mem_time_str.replace("Z", "+00:00"))
                if datetime.now(mem_time.tzinfo) - mem_time < timedelta(days=7):
                    fresh_memories.append(m)
            except Exception as e:
                print(f"Timestamp parse error: {e}")

    if fresh_memories:
        use_memory = True
        memory_context = "\n".join([m["memory"] for m in fresh_memories])
    else:
        memory_context = "No recent memory found. Use fresh web-based information if available."

    print("Memory Context------------")
    print(memory_context)

    prompt = f"""
    Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Context:
    {memory_context}

    Question: {user_input}
    """

    # Get LLM response
    manager_response = customer_bot.generate_reply(messages=[{"content": prompt, "role": "user"}])
    llm_reply = manager_response['content']
    print(manager_response)

    fallback_phrases = [
    "i do not have access",
    "i'm sorry, i cannot",
    "as an ai language model",
    "i don't know",
    "i cannot fulfill this request",
    "i am not sure",
    "i cannot access your personal information",
    "okay, i understand"
    ]

    if len(llm_reply.strip()) == 0 or any(phrase in llm_reply.lower() for phrase in fallback_phrases):
        print("INSIDE WEB SEARCH")
        web_response = web_search_agent.run(user_input).content
        llm_reply = web_response if isinstance(web_response, str) else str(web_response)

    # Save in chat
    st.session_state.chat_history.append({"role": "assistant", "content": llm_reply})
    print("LLM_REPLY-", llm_reply)

    # Save the full conversation turn to memory
    if len(llm_reply) > 20:
        conversation = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": llm_reply}
        ]
        print(f"Attempting to save to memory: {conversation}")  # Debugging line
        try:
            print(memory)
            memory.add(messages=conversation, user_id=user_id)
            print(f"Memory Added")
        except Exception as e:
            print("Memory add failed:", e)  # Error logging in case memory fails

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Group messages into Q&A pairs
paired_history = []
i = 0
while i < len(st.session_state.chat_history) - 1:
    user_msg = st.session_state.chat_history[i]
    bot_msg = st.session_state.chat_history[i + 1]
    if user_msg["role"] == "user" and bot_msg["role"] == "assistant":
        paired_history.append((user_msg, bot_msg))
        i += 2
    else:
        i += 1  # Skip if malformed pair

# Reverse for latest Q&A at the top
for user_msg, bot_msg in reversed(paired_history):
    st.markdown(f'<div class="user-msg">{user_msg["content"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">{bot_msg["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)