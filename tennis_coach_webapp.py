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

def test_players_table_connection():
    """Test if we can connect to Players table"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        response = requests.get(url, headers=headers)
        st.write(f"**Players Table Connection Test:**")
        st.write(f"Status: {response.status_code}")
        if response.status_code != 200:
            st.write(f"Error: {response.text}")
        else:
            st.write("✅ Connection successful!")
            # Show table structure
            data = response.json()
            if data.get('records'):
                st.write("**Existing records found:**", len(data['records']))
                if data['records']:
                    st.write("**Sample record fields:**")
                    sample_fields = list(data['records'][0].get('fields', {}).keys())
                    st.write(sample_fields)
        
    except Exception as e:
        st.error(f"Connection test failed: {e}")

def find_player_by_email(email: str):
    """Look up player in SSS Players table by email"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {"filterByFormula": f"{{email}} = '{email}'"}
        
        response = requests.get(url, headers=headers, params=params)
        st.write(f"**Player Lookup Debug:**")
        st.write(f"Email searched: {email}")
        st.write(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            records = response.json().get('records', [])
            st.write(f"Records found: {len(records)}")
            return records[0] if records else None
        else:
            st.write(f"Lookup error: {response.text}")
        return None
    except Exception as e:
        st.error(f"Error finding player: {e}")
        return None

def create_new_player(email: str, name: str, tennis_level: str, primary_goals: list, learning_style: str):
    """Create new player record in SSS Players table"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Only include fields that aren't computed/formula fields
        data = {
            "fields": {
                "email": email,
                "name": name,
                "tennis_level": tennis_level,
                "primary_goals": primary_goals,
                "learning_style": learning_style,
                "personality_notes": "",
                "total_sessions": 1,
                "first_session_date": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "player_status": "Active"
                # Removed last_session_date - it's a computed field
            }
        }
        
        # Debug: Show what we're sending
        st.write("**Debug - Data being sent (fixed):**")
        st.json(data)
        
        response = requests.post(url, headers=headers, json=data)
        
        # Debug: Show response
        st.write("**Debug - Response:**")
        st.write(f"Status Code: {response.status_code}")
        st.write(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            st.success("✅ Player created successfully!")
            return response.json()
        else:
            st.error(f"❌ Still getting error: {response.text}")
        return None
    except Exception as e:
        st.error(f"Error creating player: {e}")
        return None

def update_player_session_count(player_record_id: str):
    """Update player's total sessions (last_session_date is computed)"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # First get current session count
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            current_data = response.json()
            current_sessions = current_data.get('fields', {}).get('total_sessions', 0)
            
            # Update only total_sessions (last_session_date is computed)
            update_data = {
                "fields": {
                    "total_sessions": current_sessions + 1
                    # Removed last_session_date update - it's computed
                }
            }
            
            update_response = requests.patch(url, headers=headers, json=update_data)
            return update_response.status_code == 200
        return False
    except Exception as e:
        st.error(f"Error updating player: {e}")
        return False

def log_message_to_sss(player_record_id: str, session_id: str, message_order: int, role: str, content: str, chunks=None) -> bool:
    """Log message to SSS Active_Sessions table"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Calculate token count (rough estimate)
        token_count = len(content.split()) * 1.3  # Rough token estimation
        
        # Map role values to match your exact dropdown options (lowercase)
        role_value = "coach" if role == "assistant" else "player"
        
        # Convert session_id to number (remove letters, keep only digits)
        session_id_number = int(''.join(filter(str.isdigit, session_id))) if session_id else 1
        
        # Match your exact field structure
        data = {
            "fields": {
                "player_id": [player_record_id],  # Link to Players table
                "session_id": session_id_number,  # Number field
                "message_order": message_order,   # Number field
                "role": role_value,               # Single select: "player" or "coach"
                "message_content": content[:100000],  # Long text
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),  # Date
                "token_count": int(token_count),  # Number
                "session_status": "active"       # Single select: "active"
            }
        }
        
        # Debug: Show what we're sending to SSS
        st.write(f"**SSS Message Log Debug (Correct Structure):**")
        st.write(f"Player ID: {player_record_id}")
        st.write(f"Session ID: {session_id} → {session_id_number} (number)")
        st.write(f"Message #{message_order}: {role} → {role_value}")
        st.write(f"Content length: {len(content)} chars")
        st.write(f"Estimated tokens: {int(token_count)}")
        
        st.write("**Data being sent:**")
        st.json(data)
        
        response = requests.post(url, headers=headers, json=data)
        
        st.write(f"**SSS Log Response:**")
        st.write(f"Status: {response.status_code}")
        if response.status_code != 200:
            st.write(f"Error: {response.text}")
            return False
        else:
            st.write("✅ Message logged to SSS successfully!")
            return True
        
    except Exception as e:
        st.error(f"Error logging to SSS: {e}")
        return False

def test_active_sessions_connection():
    """Test if we can connect to Active_Sessions table"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        response = requests.get(url, headers=headers)
        st.write(f"**Active_Sessions Table Test:**")
        st.write(f"Status: {response.status_code}")
        if response.status_code != 200:
            st.write(f"Error: {response.text}")
        else:
            st.write("✅ Active_Sessions table accessible!")
            data = response.json()
            st.write(f"Current active messages: {len(data.get('records', []))}")
        
    except Exception as e:
        st.error(f"Active_Sessions test failed: {e}")

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
            st.session_state.player_setup_complete = False
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
    
    # Updated player setup with email collection and SSS integration
    if not st.session_state.get("player_setup_complete"):
        with st.form("player_setup"):
            st.markdown("### 🎾 Welcome to Tennis Coach AI")
            st.markdown("**Quick setup to personalize your coaching experience:**")
            
            # Email collection (required)
            player_email = st.text_input(
                "Email address *", 
                placeholder="your.email@example.com",
                help="Required for session continuity and progress tracking"
            )
            
            # Name collection  
            player_name = st.text_input(
                "Your name", 
                placeholder="e.g., Sarah, Mike, Alex...",
                help="How should your coach address you?"
            )
            
            # Tennis level
            tennis_level = st.selectbox(
                "Tennis level",
                ["Beginner", "Intermediate", "Advanced"],
                help="Helps customize coaching advice"
            )
            
            # Primary goals
            primary_goals = st.multiselect(
                "What do you want to work on?",
                ["Technique", "Fitness", "Competition", "Fun", "Mental Game"],
                default=["Technique"],
                help="Select all that apply"
            )
            
            # Learning style
            learning_style = st.selectbox(
                "Learning style",
                ["Visual", "Kinesthetic", "Analytical", "Social"],
                help="How do you learn best?"
            )
            
            if st.form_submit_button("Start Coaching Session", type="primary"):
                # Validation
                if not player_email or "@" not in player_email:
                    st.error("Please enter a valid email address.")
                elif not player_name:
                    st.error("Please enter your name.")
                else:
                    with st.spinner("Setting up your coaching session..."):
                        # Test connections first
                        test_players_table_connection()
                        test_active_sessions_connection()
                        
                        # Look up existing player
                        existing_player = find_player_by_email(player_email)
                        
                        if existing_player:
                            # Returning player
                            player_data = existing_player['fields']
                            st.session_state.player_record_id = existing_player['id']
                            st.session_state.is_returning_player = True
                            st.session_state.previous_sessions = player_data.get('total_sessions', 0)
                            
                            # Update session count
                            update_player_session_count(existing_player['id'])
                            
                            welcome_type = "returning"
                            session_info = f"This is session #{player_data.get('total_sessions', 0) + 1}"
                            
                        else:
                            # New player
                            st.write("**Creating new player profile...**")
                            new_player = create_new_player(player_email, player_name, tennis_level, primary_goals, learning_style)
                            if new_player:
                                st.session_state.player_record_id = new_player['id']
                                st.session_state.is_returning_player = False
                                st.session_state.previous_sessions = 0
                                welcome_type = "new"
                                session_info = "Welcome to your first session!"
                                st.success("✅ New player profile created successfully!")
                            else:
                                st.error("❌ Error creating player profile. Please check the debug info above.")
                                return
                        
                        # Store player info in session state
                        st.session_state.player_email = player_email
                        st.session_state.player_name = player_name
                        st.session_state.tennis_level = tennis_level
                        st.session_state.primary_goals = primary_goals
                        st.session_state.learning_style = learning_style
                        st.session_state.player_setup_complete = True
                        
                        # Initialize session variables
                        session_id = str(uuid.uuid4())[:8]
                        st.session_state.session_id = session_id
                        st.session_state.airtable_record_id = None
                        st.session_state.messages = []
                        st.session_state.conversation_log = []
                        st.session_state.message_counter = 0
                        
                        # Create session record (using old system for now)
                        airtable_record_id = create_session_record(session_id, player_name)
                        if airtable_record_id:
                            st.session_state.airtable_record_id = airtable_record_id
                        
                        # Personalized welcome message
                        if welcome_type == "returning":
                            welcome_msg = f"""👋 Welcome back, {player_name}! Great to see you again.
                            
{session_info} - I remember you're a {tennis_level.lower()} player working on {', '.join(primary_goals).lower()}. 

What would you like to focus on in today's session?"""
                        else:
                            welcome_msg = f"""👋 Hi {player_name}! I'm your tennis coach, and {session_info}
                            
I see you're a {tennis_level.lower()} player focusing on {', '.join(primary_goals).lower()}. What would you like to work on today?

I can help with technique, strategy, mental game, or any specific issues you're having on court."""
                        
                        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                        st.session_state.conversation_log = [{
                            "role": "assistant", 
                            "content": welcome_msg,
                            "timestamp": time.time()
                        }]
                        
                        # Log welcome message to SSS
                        if st.session_state.get("player_record_id"):
                            log_message_to_sss(
                                st.session_state.player_record_id,
                                session_id,
                                0,  # Welcome message is #0
                                "assistant",
                                welcome_msg
                            )
                        
                        success_msg = f"Welcome {player_name}! " + ("Returning player recognized." if welcome_type == "returning" else "New player profile created.")
                        st.success(success_msg)
                        st.rerun()
        return
    
    # Debug info (temporary - remove later)
    if st.session_state.get("player_setup_complete"):
        with st.expander("🔍 Debug: Player Info & SSS Integration"):
            st.write("Email:", st.session_state.get("player_email"))
            st.write("Name:", st.session_state.get("player_name"))
            st.write("Level:", st.session_state.get("tennis_level"))
            st.write("Goals:", st.session_state.get("primary_goals"))
            st.write("Style:", st.session_state.get("learning_style"))
            st.write("---")
            st.write("**SSS Integration:**")
            st.write("Player Record ID:", st.session_state.get("player_record_id"))
            st.write("Returning Player:", st.session_state.get("is_returning_player"))
            st.write("Previous Sessions:", st.session_state.get("previous_sessions"))
            st.write("Current Session ID:", st.session_state.get("session_id"))
    
    # Display conversation messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask your tennis coach..."):
        st.session_state.message_counter += 1
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.conversation_log.append({
            "role": "user", 
            "content": prompt,
            "timestamp": time.time()
        })
        
        # Log user message to both systems (old and SSS)
        if st.session_state.get("airtable_record_id"):
            log_message(
                st.session_state.airtable_record_id,
                st.session_state.message_counter,
                "user",
                prompt
            )
        
        # Log to SSS Active_Sessions
        if st.session_state.get("player_record_id"):
            log_message_to_sss(
                st.session_state.player_record_id,
                st.session_state.session_id,
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
                    
                    # Log assistant response to both systems
                    if st.session_state.get("airtable_record_id"):
                        log_message(
                            st.session_state.airtable_record_id,
                            st.session_state.message_counter,
                            "assistant",
                            response,
                            chunks
                        )
                    
                    # Log to SSS Active_Sessions
                    if st.session_state.get("player_record_id"):
                        log_message_to_sss(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
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
                    
                    # Log error message to both systems
                    if st.session_state.get("airtable_record_id"):
                        log_message(
                            st.session_state.airtable_record_id,
                            st.session_state.message_counter,
                            "assistant",
                            error_msg
                        )
                    
                    # Log to SSS Active_Sessions
                    if st.session_state.get("player_record_id"):
                        log_message_to_sss(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
                            st.session_state.message_counter,
                            "assistant",
                            error_msg
                        )

if __name__ == "__main__":
    main()
