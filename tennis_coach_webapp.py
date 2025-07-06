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

def create_new_player(email: str, name: str = ""):
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Use provided name, or extract from email, or leave empty for anonymous
        if name:
            player_name = name
        else:
            # Extract name from email as fallback
            player_name = email.split('@')[0].replace('.', ' ').title()
        
        data = {
            "fields": {
                "email": email,
                "name": player_name,
                "primary_goals": [],
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

def update_player_session_count(player_record_id: str):
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            current_data = response.json()
            current_sessions = current_data.get('fields', {}).get('total_sessions', 0)
            
            update_data = {
                "fields": {
                    "total_sessions": current_sessions + 1
                }
            }
            
            update_response = requests.patch(url, headers=headers, json=update_data)
            return update_response.status_code == 200
        return False
    except Exception as e:
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
        # st.error(f"DEBUG: Starting summary generation with {len(messages)} messages")
        # st.error(f"DEBUG: Sample message: {messages[0] if messages else 'None'}")
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

        # st.error("DEBUG: About to call Claude API for summary")
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            messages=[{"role": "user", "content": summary_prompt}]
        )
        # st.error("DEBUG: Claude API call successful")
        # st.error(f"DEBUG: Claude response length: {len(response.content[0].text)}")
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
        # st.error(f"DEBUG: Attempting to save summary - Player: {player_record_id}, Session: {session_number}")
        # st.error(f"DEBUG: Summary data keys: {list(summary_data.keys())}")
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
        # st.error(f"DEBUG: Airtable response code: {response.status_code}")
        # st.error(f"DEBUG: Airtable error details: {response.text}")
        return response.status_code == 200
        
    except Exception as e:
        return False

def process_completed_session(player_record_id: str, session_id: str, claude_client) -> bool:
    # st.error(f"DEBUG: process_completed_session called - START")
    try:
        # st.error(f"DEBUG: Getting messages for session {session_id}")
        messages = get_session_messages(player_record_id, session_id)
        # st.error(f"DEBUG: Retrieved {len(messages)} messages")
        
        if not messages:
            # st.error("DEBUG: No messages found - returning False")
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
def get_player_recent_summaries(player_record_id: str, limit: int = 3) -> list:
    """
    Get recent summaries for a specific player using proper Airtable linked record filtering
    """
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Session_Summaries"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        # FIXED: Use RECORD_ID() function for linked record filtering
        params = {
            "filterByFormula": f"RECORD_ID({{player_id}}) = '{player_record_id}'",
            "sort[0][field]": "session_number",
            "sort[0][direction]": "desc",
            "maxRecords": limit
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            summaries = []
            for record in records:
                fields = record.get('fields', {})
                summaries.append({
                    'session_number': fields.get('session_number', 0),
                    'session_date': fields.get('session_date', ''),
                    'technical_focus': fields.get('technical_focus', ''),
                    'mental_game_notes': fields.get('mental_game_notes', ''),
                    'homework_assigned': fields.get('homework_assigned', ''),
                    'next_session_focus': fields.get('next_session_focus', ''),
                    'key_breakthroughs': fields.get('key_breakthroughs', ''),
                    'condensed_summary': fields.get('condensed_summary', '')
                })
            return summaries
        else:
            print(f"Airtable API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Error getting recent summaries: {str(e)}")
        return []

# ENHANCED: Welcome message generation with better context
def generate_personalized_welcome_message(player_name: str, session_number: int, recent_summaries: list, is_returning: bool) -> str:
    """
    Generate a personalized welcome message based on coaching history
    """
    if not is_returning or not recent_summaries:
        return f"""👋 Hi! This is your Coach TA. Welcome to your first session!

I'm here to help you improve your tennis game. What shall we work on today?

I can help with:
• Technique (forehand, backhand, serve, volleys)
• Strategy and tactics
• Mental game and confidence
• Match preparation
• Any specific issues you're having on court

What's your main focus today? 🎾"""

    # Returning player with history
    last_session = recent_summaries[0]
    
    welcome_parts = [
        f"👋 Hi! This is your Coach TA. Great to see you back, {player_name}!",
        f"\n**This is session #{session_number}**"
    ]
    
    # Add context from last session
    if last_session.get('technical_focus'):
        welcome_parts.append(f"\n🎯 **Last session:** We worked on {last_session['technical_focus']}")
    
    if last_session.get('homework_assigned'):
        welcome_parts.append(f"\n📝 **Your homework:** {last_session['homework_assigned']}")
        welcome_parts.append("\n*How did that practice go?*")
    
    if last_session.get('next_session_focus'):
        welcome_parts.append(f"\n🎾 **Today's focus:** {last_session['next_session_focus']}")
    
    if last_session.get('key_breakthroughs'):
        welcome_parts.append(f"\n⚡ **Last breakthrough:** {last_session['key_breakthroughs']}")
    
    welcome_parts.append("\n\nReady to continue your tennis journey? What shall we work on today?")
    
    return "".join(welcome_parts)

# ENHANCED: Build conversational prompt with coaching history
def build_conversational_prompt_with_history(user_question: str, context_chunks: list, conversation_history: list, coaching_history: list = None) -> str:
    """
    Build Claude prompt including coaching history for better continuity
    """
    
    # Base coaching context
    coaching_context = """You are an expert tennis coach with deep knowledge of technique, strategy, and mental game. 
Your goal is to provide personalized, actionable advice that helps players improve systematically."""
    
    # Add coaching history if available
    if coaching_history and len(coaching_history) > 0:
        history_context = "\n\nPLAYER'S COACHING HISTORY:\n"
        for i, session in enumerate(coaching_history[:2], 1):  # Last 2 sessions
            history_context += f"\nSession {session.get('session_number', i)}:\n"
            if session.get('technical_focus'):
                history_context += f"- Technical focus: {session['technical_focus']}\n"
            if session.get('homework_assigned'):
                history_context += f"- Homework assigned: {session['homework_assigned']}\n"
            if session.get('key_breakthroughs'):
                history_context += f"- Breakthrough: {session['key_breakthroughs']}\n"
        
        coaching_context += history_context
        coaching_context += "\nUse this history to provide continuity and reference previous work when relevant."
    
    # Build the full prompt
    context_text = "\n\n".join([chunk.get('content', '') for chunk in context_chunks if chunk.get('content')])
    
    recent_conversation = ""
    if conversation_history:
        recent_messages = conversation_history[-6:]  # Last 3 exchanges
        for msg in recent_messages:
            role = "Player" if msg['role'] == 'user' else "Coach"
            recent_conversation += f"{role}: {msg['content']}\n"
    
    full_prompt = f"""{coaching_context}

RELEVANT TENNIS KNOWLEDGE:
{context_text}

RECENT CONVERSATION:
{recent_conversation}

CURRENT QUESTION: {user_question}

Provide helpful, specific tennis coaching advice. Reference previous sessions naturally when relevant. Keep responses conversational and actionable."""

    return full_prompt

# UPDATED: Main function section for player setup (replace the existing player setup section)
def setup_player_session_with_continuity(player_email: str):
    """
    Enhanced player setup with proper continuity system - CORRECTED VERSION
    """
    existing_player = find_player_by_email(player_email)
    
    if existing_player:
        # Returning player
        player_data = existing_player['fields']
        st.session_state.player_record_id = existing_player['id']
        st.session_state.is_returning_player = True
        player_name = player_data.get('name', 'there')
        session_number = player_data.get('total_sessions', 0) + 1
        
        # Load recent coaching history
        with st.spinner("Loading your coaching history..."):
            recent_summaries = get_player_recent_summaries(existing_player['id'], 2)
            st.session_state.coaching_history = recent_summaries
        
        # Generate welcome message for returning player
        if recent_summaries:
            last_session = recent_summaries[0]
            context_text = f"\n\nLast session we worked on: {last_session.get('technical_focus', 'technique practice')}"
            if last_session.get('homework_assigned'):
                context_text += f"\n\nI assigned you: {last_session.get('homework_assigned', '')}"
            if last_session.get('next_session_focus'):
                context_text += f"\n\nToday I'd like to focus on: {last_session.get('next_session_focus', '')}"
            context_text += "\n\nHow did that practice go? Ready to continue?"
        else:
            context_text = "\n\nWhat shall we work on today?"
        
        welcome_msg = f"👋 Hi! This is your Coach TA. Great to see you back, {player_name}!\n\nThis is session #{session_number}{context_text}"
        
        update_player_session_count(existing_player['id'])
        
    else:
        # New player
        new_player = create_new_player(player_email, st.session_state.get('player_name_input', ''))
        if new_player:
            st.session_state.player_record_id = new_player['id']
            st.session_state.is_returning_player = False
            st.session_state.coaching_history = []
            welcome_msg = f"👋 Hi! This is your Coach TA. Welcome to your first session!\n\nI'm here to help you improve your tennis game. What shall we work on today?\n\nI can help with technique, strategy, mental game, or any specific issues you're having on court."
        else:
            st.error("Error creating player profile. Please try again.")
            return None
    
    return welcome_msg
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

            player_name = st.text_input(
                "Name (optional)", 
                placeholder="What would you like the coach to call you?",
                help="Leave blank if you prefer to stay anonymous"
            )
            
            if st.form_submit_button("Start Coaching Session", type="primary"):
                if not player_email or "@" not in player_email:
                    st.error("Please enter a valid email address.")
                else:
                    # Store the optional name in session state
                    st.session_state.player_name_input = player_name.strip() if player_name else ""
                    with st.spinner("Setting up your coaching session..."):
                        welcome_msg = setup_player_session_with_continuity(player_email)
                        if not welcome_msg:
                            return
                        
                        st.session_state.player_email = player_email
                        st.session_state.player_setup_complete = True
                        
                        session_id = str(uuid.uuid4())[:8]
                        st.session_state.session_id = session_id
                        st.session_state.messages = []
                        st.session_state.message_counter = 0                        
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
                    # st.error(f"DEBUG: About to mark session complete - Player: {st.session_state.player_record_id}")
                    session_marked = mark_session_completed(
                        st.session_state.player_record_id,
                        st.session_state.session_id
                    )
                    # st.error(f"DEBUG: Session marked result: {session_marked}")
                    if session_marked:
                        # st.error("DEBUG: Entering summary generation block")
                        pass
                    else:
                        # st.error("DEBUG: Session marked returned False - no summary generation")
                        pass
                    if session_marked:
                        st.success("✅ Session marked as completed!")
                        
                        
                        # Generate session summary
                        with st.spinner("🧠 Generating session summary..."):
                            st.error(f"DEBUG: player_record_id value = '{st.session_state.player_record_id}'")
                            # st.error(f"DEBUG: About to process session - Player: {st.session_state.player_record_id}, Session: {st.session_state.session_id}")
                            summary_created = process_completed_session(
                                st.session_state.player_record_id,
                                st.session_state.session_id,
                                claude_client
                            )
                            # st.error(f"DEBUG: Summary result: {summary_created}")
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
                    coaching_history = st.session_state.get('coaching_history', [])
                    full_prompt = build_conversational_prompt_with_history(
                        prompt, 
                        chunks, 
                        st.session_state.messages[:-1],
                        coaching_history
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
