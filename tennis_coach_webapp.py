import streamlit as st
import os
import json
from typing import List, Dict
import time

try:
    from pinecone import Pinecone
    import openai
    import anthropic
    import requests
    import uuid
    import platform
except ImportError as e:
    st.error(f"Missing package: {e}")
    st.stop()

@st.cache_resource
def setup_connections():
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index(st.secrets["PINECONE_INDEX_NAME"])
        claude_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        return index, claude_client
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None, None

def get_embedding(text: str) -> List[float]:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        st.write(f"Debug - Using OpenAI key ending in: ...{api_key[-4:]}")
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []
def extract_array_value(metadata_field):
    if not metadata_field:
        return "Not specified"
    if isinstance(metadata_field, list):
        if len(metadata_field) > 0:
            for item in metadata_field:
                if item and str(item).strip():
                    return str(item).strip()
        return "Not specified"
    if isinstance(metadata_field, str):
        if metadata_field.startswith('[') and metadata_field.endswith(']'):
            cleaned = metadata_field.strip('[]').replace('"', '').replace("'", "")
            cleaned = ' '.join(cleaned.split())
            return cleaned if cleaned else "Not specified"
    return str(metadata_field).strip() if metadata_field else "Not specified"

def query_pinecone(index, question: str, top_k: int = 3) -> List[Dict]:
    try:
        question_vector = get_embedding(question)
        if not question_vector:
            return []
        results = index.query(
            vector=question_vector,
            top_k=top_k,
            include_metadata=True
        )
        chunks = [
            {
                'text': match.metadata.get('text_preview', ''),
                'score': match.score,
                'source': match.metadata.get('source_url', 'Unknown'),
                'topics': match.metadata.get('tennis_topics', ''),
                'skill_level': extract_array_value(match.metadata.get('skill_level')),
                'coaching_style': extract_array_value(match.metadata.get('coaching_style'))
            }
            for match in results.matches
        ]
        return chunks
    except Exception as e:
        st.error(f"Pinecone query error: {e}")
        return []

def build_conversational_prompt(question: str, chunks: List[Dict], conversation_history: List[Dict]) -> str:
    context_sections = []
    for i, chunk in enumerate(chunks):
        context_sections.append(f"""
Resource {i+1}:
Topics: {chunk['topics']}
Level: {chunk['skill_level']}
Style: {chunk['coaching_style']}
Content: {chunk['text']}
""")
    context_text = "\n".join(context_sections)
    history_text = ""
    if conversation_history:
        history_text = "\nPrevious conversation:\n"
        for msg in conversation_history[-6:]:
            role = "Player" if msg['role'] == 'user' else "Coach"
            history_text += f"{role}: {msg['content']}\n"
    return f"""You are a professional tennis coach providing REMOTE coaching advice through chat. The player is not physically with you, so focus on guidance they can apply on their own.

Guidelines:
- CRITICAL: Keep responses very short - maximum 3-4 sentences (phone screen length)
- Focus on ONE specific tip or correction per response
- Give advice for SOLO practice or general technique improvement
- Ask one engaging follow-up question to continue the conversation
- Use encouraging, supportive tone
- Be direct and actionable
- DO NOT suggest feeding balls, court positioning, or activities requiring a coach present
- Focus on: technique tips, solo drills, mental approach, general strategy

{history_text}

Professional Coaching Resources:
{context_text}

Current Player Question: "{question}"

Respond as their remote tennis coach with a SHORT, focused response:"""

def query_claude(client, prompt: str) -> str:
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating coaching response: {e}"
def create_session_record(session_id: str, tester_name: str) -> str:
    try:
        url = f"https://api.airtable.com/v0/{st.secrets['AIRTABLE_BASE_ID']}/Test_Sessions"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        device_info = f"{platform.system()} - {platform.processor()}"
        data = {
            "fields": {
                "session_id": session_id,
                "tester_name": tester_name,
                "total_messages": 0,
                "device_info": device_info
            }
        }
        print(f"Creating session: {session_id} for {tester_name}")
        response = requests.post(url, headers=headers, json=data)
        print(f"Session creation response: {response.status_code} - {response.text}")
        if response.status_code == 200:
            record_data = response.json()
            record_id = record_data['id']
            print(f"Session record ID: {record_id}")
            return record_id
        else:
            return None
    except Exception as e:
        print(f"Error creating session: {e}")
        st.error(f"Error creating session: {e}")
        return None

def log_message(session_record_id: str, message_order: int, role: str, content: str, chunks=None) -> bool:
    try:
        url = f"https://api.airtable.com/v0/{st.secrets['AIRTABLE_BASE_ID']}/Conversation_Log"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        resource_details = ""
        resources_count = 0
        if chunks:
            resources_count = len(chunks)
            resource_details = "\n".join([
                f"Resource {i+1}: {chunk['topics']} (Score: {chunk['score']:.3f}) - {chunk['skill_level']}"
                for i, chunk in enumerate(chunks)
            ])
        data = {
            "fields": {
                "session_id": [session_record_id],
                "message_order": message_order,
                "role": role,
                "message_content": content[:100000],
                "coaching_resources_used": resources_count,
                "resource_details": resource_details
            }
        }
        print(f"Logging message {message_order}: {role} - {content[:50]}...")
        response = requests.post(url, headers=headers, json=data)
        print(f"Message log response: {response.status_code} - {response.text[:200]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"Error logging message: {e}")
        st.error(f"Error logging message: {e}")
        return False

def update_session_stats(session_id: str, total_messages: int) -> bool:
    try:
        url = f"https://api.airtable.com/v0/{st.secrets['AIRTABLE_BASE_ID']}/Test_Sessions"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"
        }
        params = {
            "filterByFormula": f"{{session_id}} = '{session_id}'"
        }
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                record_id = records[0]['id']
                update_url = f"{url}/{record_id}"
                update_data = {
                    "fields": {
                        "total_messages": total_messages,
                        "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    }
                }
                update_response = requests.patch(update_url, headers=headers, json=update_data)
                return update_response.status_code == 200
        return False
    except Exception as e:
        print(f"Error updating session: {e}")
        return False
        def main():
    st.set_page_config(
        page_title="Tennis Coach AI",
        page_icon="🎾",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🎾 Tennis Coach AI")
    st.markdown("*Your personal tennis coaching assistant*")
    st.markdown("---")
    
    with st.spinner("Connecting to tennis coaching database..."):
        index, claude_client = setup_connections()
    
    if not index or not claude_client:
        st.error("Failed to connect to coaching systems. Please check API keys.")
        st.stop()
    
    with st.sidebar:
        st.header("🔧 Admin Controls")
        top_k = st.slider("Coaching resources", 1, 8, 3)
        
        if st.button("🔄 New Session"):
            st.session_state.messages = []
            st.session_state.conversation_log = []
            st.rerun()
        
        if 'conversation_log' in st.session_state and st.session_state.conversation_log:
            st.markdown(f"**Session messages:** {len(st.session_state.conversation_log)}")
            
            with st.expander("📋 Full Session Log"):
                for i, entry in enumerate(st.session_state.conversation_log):
                    st.markdown(f"**Message {i+1}:** {entry['role']}")
                    st.markdown(f"*Content:* {entry['content'][:100]}...")
                    if 'chunks' in entry:
                        st.markdown(f"*Sources used:* {len(entry['chunks'])} resources")
                        for j, chunk in enumerate(entry['chunks']):
                            st.markdown(f"  - Resource {j+1}: {chunk['topics']} (score: {chunk['score']:.3f})")
    
    if "messages" not in st.session_state:
        session_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = session_id
        st.session_state.airtable_record_id = None
        st.session_state.messages = []
        st.session_state.conversation_log = []
        st.session_state.message_counter = 0
        
        if "tester_name" not in st.session_state:
            st.session_state.tester_name = None
        
        welcome_msg = """👋 Hi! I'm your tennis coach. What would you like to work on today?

I can help with technique, strategy, mental game, or any specific issues you're having on court."""
        
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
        st.session_state.conversation_log.append({
            "role": "assistant", 
            "content": welcome_msg,
            "timestamp": time.time()
        })
    
    if not st.session_state.get("tester_name"):
        with st.form("tester_info"):
            st.markdown("**Quick setup:** What's your name? (optional, for session tracking)")
            tester_name = st.text_input("Your name", placeholder="e.g., Alex, Sarah, Coach Mike...")
            if st.form_submit_button("Start Coaching"):
                st.session_state.tester_name = tester_name if tester_name else "Anonymous"
                airtable_record_id = create_session_record(st.session_state.session_id, st.session_state.tester_name)
                if airtable_record_id:
                    st.session_state.airtable_record_id = airtable_record_id
                    st.rerun()
                else:
                    st.error("Failed to create session record. Please try again.")
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your tennis coach..."):
        st.session_state.message_counter += 1
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_log.append({
            "role": "user", 
            "content": prompt,
            "timestamp": time.time()
        })
        
        if st.session_state.get("airtable_record_id"):
            log_message(
                st.session_state.airtable_record_id,
                st.session_state.message_counter,
                "user",
                prompt
            )
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Coach is thinking..."):
                chunks = query_pinecone(index, prompt, top_k)
                
                if chunks:
                    full_prompt = build_conversational_prompt(
                        prompt, 
                        chunks, 
                        st.session_state.messages[:-1]
                    )
                    
                    response = query_claude(claude_client, full_prompt)
                    
                    st.markdown(response)
                    
                    st.session_state.message_counter += 1
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    st.session_state.conversation_log.append({
                        "role": "assistant", 
                        "content": response,
                        "chunks": chunks,
                        "timestamp": time.time(),
                        "prompt_used": full_prompt
                    })
                    
                    if st.session_state.get("airtable_record_id"):
                        log_message(
                            st.session_state.airtable_record_id,
                            st.session_state.message_counter,
                            "assistant",
                            response,
                            chunks
                        )
                    
                    update_session_stats(st.session_state.session_id, st.session_state.message_counter)
                    
                else:
                    error_msg = "Could you rephrase that? I want to give you the best coaching advice possible."
                    st.markdown(error_msg)
                    st.session_state.message_counter += 1
                    
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.session_state.conversation_log.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": time.time()
                    })
                    
                    if st.session_state.get("airtable_record_id"):
                        log_message(
                            st.session_state.airtable_record_id,
                            st.session_state.message_counter,
                            "assistant",
                            error_msg
                        )

if __name__ == "__main__":
    main()
