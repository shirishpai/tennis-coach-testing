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
    import time
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
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
            if "529" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
            return f"Error generating coaching response: {e}"

def find_player_by_email(email: str):
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {"filterByFormula": f"{{email}} = '{email}'"}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            return records[0] if records else None
        return None
    except Exception as e:
        return None

def create_new_player(email: str):
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "fields": {
                "email": email,
                "name": "",
                "tennis_level": "Unknown",
                "primary_goals": [],
                "learning_style": "Unknown",
                "personality_notes": "",
                "total_sessions": 1,
                "first_session_date": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "player_status": "Active"
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Exception details: {str(e)}")
        return None

def detect_session_end(message_content: str) -> bool:
    goodbye_phrases = [
        "thanks", "thank you", "bye", "goodbye", "see you", "done", 
        "that's all", "finished", "end session", "stop", "quit",
        "done for today", "good session", "catch you later", "later",
        "gotta go", "have to go", "thanks coach", "thank you coach"
    ]
    
    message_lower = message_content.lower().strip()
    
    for phrase in goodbye_phrases:
        if phrase in message_lower:
            return True
    
    if len(message_lower.split()) <= 3:
        ending_words = ["thanks", "bye", "done", "good", "great"]
        if any(word in message_lower for word in ending_words):
            return True
    
    return False

def mark_session_completed(player_record_id: str, session_id: str) -> bool:
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        session_id_number = int(''.join(filter(str.isdigit, session_id))) if session_id else 1
        
        params = {
            "filterByFormula": f"AND({{session_id}} = {session_id_number}, {{session_status}} = 'active')"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            
            update_headers = {
                "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
                "Content-Type": "application/json"
            }
            
            for record in records:
                record_id = record['id']
                update_url = f"{url}/{record_id}"
                update_data = {
                    "fields": {
                        "session_status": "completed"
                    }
                }
                
                requests.patch(update_url, headers=update_headers, json=update_data)
            
            return len(records) > 0
        
        return False
    except Exception as e:
        return False

def get_session_messages(player_record_id: str, session_id: str) -> list:
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        session_id_number = int(''.join(filter(str.isdigit, session_id))) if session_id else 1
        
        params = {
            "filterByFormula": f"{{session_id}} = {session_id_number}",
            "sort[0][field]": "message_order",
            "sort[0][direction]": "asc"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            messages = []
            for record in records:
                fields = record.get('fields', {})
                messages.append({
                    'role': fields.get('role', ''),
                    'content': fields.get('message_content', ''),
                    'order': fields.get('message_order', 0)
                })
            return messages
        return []
    except Exception as e:
        return []
def generate_session_summary(messages: list, claude_client) -> dict:
    try:
        st.error(f"DEBUG: Starting summary generation with {len(messages)} messages")
        st.error(f"DEBUG: Sample message: {messages[0] if messages else 'None'}")
        conversation_text = ""
        for msg in messages:
            role_label = "Player" if msg['role'] == 'player' else "Coach"
            conversation_text += f"{role_label}: {msg['content']}\n\n"
        
        summary_prompt = f"""Analyze this tennis coaching session and extract key information. The session is between a coach and player working on tennis improvement.

CONVERSATION:
{conversation_text}

Please analyze and provide a structured summary with these exact sections:

TECHNICAL_FOCUS: What specific tennis techniques were discussed or worked on? (e.g., forehand grip, serve motion, backhand slice)

MENTAL_GAME: Any mindset, confidence, or mental approach topics covered? (e.g., staying calm, visualization, match preparation)

HOMEWORK_ASSIGNED: What practice tasks or exercises were given to the player? (e.g., wall hitting, shadow swings, specific drills)

NEXT_SESSION_FOCUS: Based on this session, what should be the priority for the next coaching session?

KEY_BREAKTHROUGHS: Any important progress moments, "aha" moments, or skill improvements noted?

CONDENSED_SUMMARY: Write a concise 200-300 token summary capturing the essence of this coaching session, focusing on what was learned and accomplished.

Format your response exactly like this:
TECHNICAL_FOCUS: [your analysis]
MENTAL_GAME: [your analysis]  
HOMEWORK_ASSIGNED: [your analysis]
NEXT_SESSION_FOCUS: [your analysis]
KEY_BREAKTHROUGHS: [your analysis]
CONDENSED_SUMMARY: [your analysis]"""

        st.error("DEBUG: About to call Claude API for summary")
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        st.error("DEBUG: Claude API call successful")
        st.error(f"DEBUG: Claude response length: {len(response.content[0].text)}")
        summary_text = response.content[0].text
        
        summary_data = {}
        current_section = None
        current_content = []
        
        for line in summary_text.split('\n'):
            line = line.strip()
            if line.startswith('TECHNICAL_FOCUS:'):
                if current_section:
                    summary_data[current_section] = ' '.join(current_content).strip()
                current_section = 'technical_focus'
                current_content = [line.replace('TECHNICAL_FOCUS:', '').strip()]
            elif line.startswith('MENTAL_GAME:'):
                if current_section:
                    summary_data[current_section] = ' '.join(current_content).strip()
                current_section = 'mental_game_notes'
                current_content = [line.replace('MENTAL_GAME:', '').strip()]
            elif line.startswith('HOMEWORK_ASSIGNED:'):
                if current_section:
                    summary_data[current_section] = ' '.join(current_content).strip()
                current_section = 'homework_assigned'
                current_content = [line.replace('HOMEWORK_ASSIGNED:', '').strip()]
            elif line.startswith('NEXT_SESSION_FOCUS:'):
                if current_section:
                    summary_data[current_section] = ' '.join(current_content).strip()
                current_section = 'next_session_focus'
                current_content = [line.replace('NEXT_SESSION_FOCUS:', '').strip()]
            elif line.startswith('KEY_BREAKTHROUGHS:'):
                if current_section:
                    summary_data[current_section] = ' '.join(current_content).strip()
                current_section = 'key_breakthroughs'
                current_content = [line.replace('KEY_BREAKTHROUGHS:', '').strip()]
            elif line.startswith('CONDENSED_SUMMARY:'):
                if current_section:
                    summary_data[current_section] = ' '.join(current_content).strip()
                current_section = 'condensed_summary'
                current_content = [line.replace('CONDENSED_SUMMARY:', '').strip()]
            elif line and current_section:
                current_content.append(line)
        
        if current_section:
            summary_data[current_section] = ' '.join(current_content).strip()
        
        return summary_data
        
    except Exception as e:
        st.error(f"DEBUG: Summary generation failed with error: {str(e)}")
        return {
            'technical_focus': 'Summary generation failed',
            'mental_game_notes': '',
            'homework_assigned': '',
            'next_session_focus': 'Continue working on tennis fundamentals',
            'key_breakthroughs': '',
            'condensed_summary': 'Coaching session completed but summary generation encountered an error.'
        }

def save_session_summary(player_record_id: str, session_number: int, summary_data: dict, original_message_count: int) -> bool:
    try:
        st.error(f"DEBUG: Attempting to save summary - Player: {player_record_id}, Session: {session_number}")
        st.error(f"DEBUG: Summary data keys: {list(summary_data.keys())}")
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Session_Summaries"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        original_tokens = original_message_count * 50
        summary_tokens = len(summary_data.get('condensed_summary', '').split()) * 1.3
        token_savings = max(0, original_tokens - summary_tokens)
        
        data = {
            "fields": {
                "player_id": [player_record_id],
                "session_number": session_number,
                "session_date": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "technical_focus": summary_data.get('technical_focus', '')[:1000],
                "mental_game_notes": summary_data.get('mental_game_notes', '')[:1000],
                "homework_assigned": summary_data.get('homework_assigned', '')[:1000], 
                "next_session_focus": summary_data.get('next_session_focus', '')[:1000],
                "key_breakthroughs": summary_data.get('key_breakthroughs', '')[:1000],
                "condensed_summary": summary_data.get('condensed_summary', '')[:2000],
                "original_msg_count": original_message_count,
                "token_cost_saved": round(token_savings, 2)
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        st.error(f"DEBUG: Airtable response code: {response.status_code}")
        st.error(f"DEBUG: Airtable error details: {response.text}")
        return response.status_code == 200
        
    except Exception as e:
        return False

def process_completed_session(player_record_id: str, session_id: str, claude_client) -> bool:
    try:
        st.error(f"DEBUG: Getting messages for session {session_id}")
        messages = get_session_messages(player_record_id, session_id)
        st.error(f"DEBUG: Retrieved {len(messages)} messages")
        
        if not messages:
            st.error("DEBUG: No messages found - returning False")
            return False
        
        summary_data = generate_session_summary(messages, claude_client)
        
        player_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        player_response = requests.get(player_url, headers=headers)
        if player_response.status_code == 200:
            player_data = player_response.json()
            session_number = player_data.get('fields', {}).get('total_sessions', 1)
        else:
            session_number = 1
        
        summary_saved = save_session_summary(
            player_record_id, 
            session_number, 
            summary_data, 
            len(messages)
        )
        
        return summary_saved
        
    except Exception as e:
        return False

def log_message_to_sss(player_record_id: str, session_id: str, message_order: int, role: str, content: str, chunks=None) -> bool:
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        token_count = len(content.split()) * 1.3
        role_value = "coach" if role == "assistant" else "player"
        session_id_number = int(''.join(filter(str.isdigit, session_id))) if session_id else 1
        
        data = {
            "fields": {
                "player_id": [player_record_id],
                "session_id": session_id_number,
                "message_order": message_order,
                "role": role_value,
                "message_content": content[:100000],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "token_count": int(token_count),
                "session_status": "active"
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 200
        
    except Exception as e:
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
            st.session_state.player_setup_complete = False
            st.rerun()
    
    if not st.session_state.get("player_setup_complete"):
        with st.form("player_setup"):
            st.markdown("### 🎾 Welcome to Tennis Coach AI")
            st.markdown("**Quick setup:**")
            
            player_email = st.text_input(
                "Email address", 
                placeholder="your.email@example.com",
                help="Required for session continuity and progress tracking"
            )
            
            if st.form_submit_button("Start Coaching Session", type="primary"):
                if not player_email or "@" not in player_email:
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner("Setting up your coaching session..."):
                        existing_player = find_player_by_email(player_email)
                        
                        if existing_player:
                            player_data = existing_player['fields']
                            st.session_state.player_record_id = existing_player['id']
                            st.session_state.is_returning_player = True
                            player_name = player_data.get('name', 'there')
                            welcome_type = "returning"
                            session_info = f"This is session #{player_data.get('total_sessions', 0) + 1}"
                        else:
                            new_player = create_new_player(player_email)
                            if new_player:
                                st.session_state.player_record_id = new_player['id']
                                st.session_state.is_returning_player = False
                                player_name = "there"
                                welcome_type = "new"
                                session_info = "Welcome to your first session!"
                            else:
                                st.error("Error creating player profile. Please try again.")
                                return
                        
                        st.session_state.player_email = player_email
                        st.session_state.player_setup_complete = True
                        
                        session_id = str(uuid.uuid4())[:8]
                        st.session_state.session_id = session_id
                        st.session_state.messages = []
                        st.session_state.message_counter = 0
                        
                        if welcome_type == "returning":
                            welcome_msg = f"👋 Hi! This is your Coach TA. Great to see you back, {player_name}!\n\n{session_info} - What shall we work on today?"
                        else:
                            welcome_msg = f"👋 Hi! This is your Coach TA. {session_info}\n\nI'm here to help you improve your tennis game. What shall we work on today?\n\nI can help with technique, strategy, mental game, or any specific issues you're having on court."
                        
                        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                        
                        if st.session_state.get("player_record_id"):
                            log_message_to_sss(
                                st.session_state.player_record_id,
                                session_id,
                                0,
                                "assistant",
                                welcome_msg
                            )
                        
                        st.success(f"Welcome! Ready to start your coaching session.")
                        st.rerun()
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask your tennis coach..."):
        if detect_session_end(prompt):
            st.session_state.session_ending = True
        
        st.session_state.message_counter += 1
        
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
        
        # If session is ending, provide closing response and mark as completed
        if st.session_state.get("session_ending"):
            with st.chat_message("assistant"):
                closing_response = "Great session today! I've saved our progress and I'll remember what we worked on. Keep practicing those techniques, and I'll be here whenever you need coaching support. Take care! 🎾"
                st.markdown(closing_response)
                
                # Log closing response
                st.session_state.message_counter += 1
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": closing_response
                })
                
                if st.session_state.get("player_record_id"):
                    log_message_to_sss(
                        st.session_state.player_record_id,
                        st.session_state.session_id,
                        st.session_state.message_counter,
                        "assistant",
                        closing_response
                    )
                
                # Mark session as completed
                if st.session_state.get("player_record_id"):
                    st.error(f"DEBUG: About to mark session complete - Player: {st.session_state.player_record_id}")
                    session_marked = mark_session_completed(
                        st.session_state.player_record_id,
                        st.session_state.session_id
                    )
                    st.error(f"DEBUG: Session marked result: {session_marked}")
                    if session_marked:
                        st.error("DEBUG: Entering summary generation block")
                    else:
                        st.error("DEBUG: Session marked returned False - no summary generation")
                    if session_marked:
                        st.success("✅ Session marked as completed!")
                        
                        # Generate session summary
                        # Generate session summary
                        with st.spinner("🧠 Generating session summary..."):
                            st.error(f"DEBUG: About to process session - Player: {st.session_state.player_record_id}, Session: {st.session_state.session_id}")
                            summary_created = process_completed_session(
                                st.session_state.player_record_id,
                                st.session_state.session_id,
                                claude_client
                            )
                            st.error(f"DEBUG: Summary result: {summary_created}")
                            summary_created = process_completed_session(
                                st.session_state.player_record_id,
                                st.session_state.session_id,
                                claude_client
                            )
                            if summary_created:
                                st.success("📝 Session summary generated and saved!")
                            else:
                                st.warning("⚠️ Session completed but summary generation had issues.")
                
                # Show session end message
                st.success("🎾 **Session Complete!** Thanks for training with Coach TA today.")
                if st.button("🔄 Start New Session", type="primary"):
                    for key in list(st.session_state.keys()):
                        if key not in ['player_email', 'player_record_id']:
                            del st.session_state[key]
                    st.rerun()
                return
        
        # Normal message processing (not ending)
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
                    
                else:
                    error_msg = "Could you rephrase that? I want to give you the best coaching advice possible."
                    st.markdown(error_msg)
                    st.session_state.message_counter += 1
                    
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
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
