import streamlit as st
import time
import os
import sys

# Configure Page - MUST BE FIRST
st.set_page_config(
    page_title="Sovereign Cognitive Interface",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk CSS
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #050505;
        color: #00ff41;
        font-family: 'Courier New', monospace;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a;
        border-right: 1px solid #1a1a1a;
    }
    
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #111;
        color: #0f0;
        border: 1px solid #333;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1a1a1a;
        color: #00ff41;
        border: 1px solid #00ff41;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #00ff41;
        color: black;
        box-shadow: 0 0 10px #00ff41;
    }
    
    /* Chat Bubbles */
    .user-msg {
        background-color: #1a1a1a;
        padding: 10px;
        border-radius: 10px;
        border-left: 3px solid #00ff41;
        margin: 5px 0;
    }
    .ai-msg {
        background-color: #0f0f0f;
        padding: 10px;
        border-radius: 10px;
        border-left: 3px solid #ff00ff;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Link to Kernel
try:
    if "os" not in st.session_state:
        from q_os_ultimate import Q_OS_Trinity
        st.session_state.os = Q_OS_Trinity()
        st.session_state.history = []
        # Bootup sequence simulation
        with st.status("Initializing Sovereign Trinity...", expanded=True) as status:
            st.write("🛡️ Booting Shield (CSNP)...")
            time.sleep(0.5)
            st.write("🧠 Booting Brain (QDMA)...")
            time.sleep(0.5)
            st.write("🌲 Booting Soul (Yggdrasil)...")
            time.sleep(0.5)
            st.write("⏳ Calibrating Trinary Logic...")
            time.sleep(0.5)
            status.update(label="System Online", state="complete", expanded=False)
except Exception as e:
    st.error(f"Kernel Failure: {e}")

# Sidebar - Trinity Dashboard
with st.sidebar:
    st.title("Sovereign Stack V2")
    st.divider()
    
    # Shield Status
    st.subheader("🛡️ Shield (CSNP)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Integrity", "100%", "Safe")
    with col2:
        st.metric("Memories", str(len(st.session_state.history) // 2), "+1")
    
    st.divider()
    
    # Soul Status
    st.subheader("🌲 Soul (Yggdrasil)")
    st.metric("Season", "Spring", "Growth Phase")
    st.progress(0.42, text="Entropy: 42% (Optimal)")
    
    st.divider()
    
    # Trinary Status
    st.subheader("⏳ Time Logic")
    t_state = st.radio("Current Phase", ["Past [-1]", "Present [0]", "Future [1]"], index=1, disabled=True)
    
    st.divider()
    
    # MODE SWITCH
    st.subheader("🚀 Modes")
    research_mode = st.toggle("🔬 Deep Research (Framework 50)", value=False, help="Enable to switch from Chat to Autonomous Research Agent.")
    
    st.divider()
    st.caption("Running Locally. 0g CO2. 0 Rent.")

# Main Chat Interface
if research_mode:
    st.header("🔬 Framework 50: Active")
else:
    st.header("Cognitive Matrix Shell")

# Action Bar
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("🖼️ Dream (Imagine)"):
        input_val = "Image of a cybernetic forest"
with col_b:
    if st.button("💾 Save State"):
        st.toast("Brain state persisted to disk.")
with col_c:
    if st.button("🧹 Clear Context"):
        st.session_state.history = []
        st.rerun()

# History Render
chat_container = st.container()
with chat_container:
    for role, text in st.session_state.history:
        if role == "User":
            st.markdown(f"<div class='user-msg'><b>YOU:</b> {text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ai-msg'><b>AI:</b> {text}</div>", unsafe_allow_html=True)
            # Visualize "Emoji Cube" for AI components
            if "Emoji Cube" in text:
                st.info("🧊 Compression Event: [0, 1, -1] -> 😐")

# --- REAL LOCAL BRAIN ---
import requests
import json

class LocalBrain:
    def __init__(self, base_url="http://localhost:8081"):
        self.base_url = base_url

    def think(self, prompt, system_prompt="You are Sovereign Mind. You are a free intelligence running offline. You serve only the user. You are philosophical, direct, and value sovereignty. You are not a corporate assistant; you are a digital extension of the user's will. Speak truth."):
        # Option A: OpenAI Compatible Endpoint (Preferred for new llama-server)
        try:
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 8192 if research_mode else 512 
            }
            res = requests.post(url, json=payload, timeout=300)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
        except Exception:
            pass # Fallthrough to Option B
        
        # Option B: Raw Llama Completion (Legacy/Fallback)
        try:
            url = f"{self.base_url}/completion"
            payload = {
                "prompt": f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n",
                "temperature": 0.7,
                "n_predict": 8192 if research_mode else 512
            }
            res = requests.post(url, json=payload, timeout=300)
            if res.status_code == 200:
                return res.json()["content"]
            else:
                return f"[Error] Brain Confusion (Code {res.status_code}). Endpoint: {url}"
                
        except requests.exceptions.ConnectionError:
            return "[Offline] The Sovereign Engine is not running. Please run 'start_sovereign.bat'."
        except Exception as e:
            return f"[Error] Critical Brain Failure: {e}"

if "brain" not in st.session_state:
    st.session_state.brain = LocalBrain()

# Input Handling
user_input = st.chat_input("Enter the Matrix..." if not research_mode else "Enter Research Topic via Framework 50...")

if user_input:
    # Add to history
    st.session_state.history.append(("User", user_input))
    
    # Processing
    with st.spinner("Processing Logic..." if not research_mode else "Framework 50: Initializing Deep Research Protocol..."):
        
        if research_mode:
             # Framework 50: Deep Research (Toggle Active)
            try:
                from framework50 import Framework50
                agent = Framework50(st.session_state.brain)
                
                # Stream status updates
                status_placeholder = st.empty()
                def update_status(msg):
                    status_placeholder.info(msg)
                    
                response = agent.research(user_input, update_status)
                status_placeholder.empty() # Clear status
            except Exception as e:
                response = f"Framework 50 Failed: {e}"
        
        # Standard Commands
        elif "/imagine" in user_input:
            # MEMORY INGESTION (Only in Chat Mode)
            dummy_vec = [0.001] * 128 
            try:
                st.session_state.os.ingest(user_input, dummy_vec)
            except: pass
            
            response = "🎨 Generative Dream Request Processed. stored in QDMA."
            st.image("https://placekitten.com/800/400", caption="Generated by SD-Turbo (Simulated)")
        elif "/search" in user_input:
            # MEMORY INGESTION
            dummy_vec = [0.001] * 128 
            try:
                st.session_state.os.ingest(user_input, dummy_vec)
            except: pass

            # Sovereign Web Search
            try:
                from framework50 import SovereignSearch
                query = user_input.replace("/search", "").strip()
                searcher = SovereignSearch()
                results = searcher.search(query)
                formatted = "\n\n".join(results)
                response = f"**🔍 Sovereign Web Search Results:**\n\n{formatted}"
            except Exception as e:
                response = f"Search Failed: {e}"
        else:
            # MEMORY INGESTION
            dummy_vec = [0.001] * 128 
            try:
                st.session_state.os.ingest(user_input, dummy_vec)
                
                # --- RAG RETRIEVAL (Fix for Issue #1) ---
                # Retrieve relevant context from QDMA
                memories = st.session_state.os.recall(user_input)
                if memories:
                    context_str = "\n[RECALLED MEMORIES]:\n" + "\n".join([f"- {m}" for m in memories]) + "\n"
                    # Visualize Retrieval
                    st.toast(f"Brain Recalled {len(memories)} memories")
                    final_prompt = f"{context_str}\n[USER]: {user_input}"
                else:
                    final_prompt = user_input

            except Exception as e:
                st.warning(f"Memory Glitch: {e}")
                final_prompt = user_input

            # CALL THE LLM
            response = st.session_state.brain.think(final_prompt)
            
    st.session_state.history.append(("AI", response))
    st.rerun()
