import streamlit as st
import httpx
import asyncio
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# Backend URL
BACKEND_URL = "http://backend:8000"

def main():
    st.title("🤖 Financial AI Agent Chat")
    st.markdown("Chat with your AI agent. Ask about currency rates or anything else!")
    
    # Initialize session state for chat history and session ID
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "tool_used" in message and message["tool_used"]:
                st.caption(f"🔧 Tool used: {message['tool_used']}")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call backend API with session ID
                    request_data = {"message": prompt}
                    if st.session_state.session_id:
                        request_data["session_id"] = st.session_state.session_id
                        
                    response = httpx.post(
                        f"{BACKEND_URL}/chat",
                        json=request_data,
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        ai_response = data["response"]
                        tool_used = data.get("tool_used")
                        session_id = data.get("session_id")
                        
                        # Store the session ID for future requests
                        if session_id:
                            st.session_state.session_id = session_id
                        
                        st.markdown(ai_response)
                        
                        if tool_used:
                            st.caption(f"🔧 Tool used: {tool_used}")
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": ai_response,
                            "tool_used": tool_used
                        })
                    else:
                        error_msg = f"Error: {response.status_code} - {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"Sorry, I encountered an error: {error_msg}"
                        })
                        
                except httpx.ConnectError:
                    error_msg = "Cannot connect to the backend. Please make sure the backend service is running."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

    with st.sidebar:
        st.header("Financial AI Agent")
        
        st.header("🛠️ Available Tools")
        st.markdown("""
        - **Currency Rates**: Real-time exchange rates from National Bank of Ukraine
        - More tools to come
        """)

        st.header("💬 Session Management")
        if st.session_state.session_id:
            st.success(f"Active Session: `{st.session_state.session_id[:8]}...`")
            st.caption(f"Messages: {len(st.session_state.messages)}")
        else:
            st.info("No active session")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("New Session"):
                st.session_state.session_id = None
                st.session_state.messages = []
                st.rerun()
        
        # Session History Management
        if st.session_state.session_id:
            if st.button("📊 View Server History"):
                try:
                    response = httpx.get(f"{BACKEND_URL}/chat/history/{st.session_state.session_id}", timeout=10.0)
                    if response.status_code == 200:
                        data = response.json()
                        st.json(data)
                    else:
                        st.error("Failed to fetch server history")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("🗑️ Clear Server History"):
                try:
                    response = httpx.delete(f"{BACKEND_URL}/chat/history/{st.session_state.session_id}", timeout=10.0)
                    if response.status_code == 200:
                        st.success("Server history cleared!")
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.rerun()
                    else:
                        st.error("Failed to clear server history")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Backend status check
        st.header("🔗 Backend Status")
        try:
            response = httpx.get(f"{BACKEND_URL}/health", timeout=5.0)
            if response.status_code == 200:
                st.success("✅ Backend Connected")
                data = response.json()
                st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")
            else:
                st.error("❌ Backend Error")
        except:
            st.error("❌ Backend Disconnected")

if __name__ == "__main__":
    main()