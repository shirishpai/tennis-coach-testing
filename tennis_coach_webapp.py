import streamlit as st
import os
import json
from typing import List, Dict
import time
import pandas as pd
from datetime import datetime

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
        # Normalize email to lowercase
        email = email.lower().strip()
        
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

def update_player_info(player_id: str, name: str = "", tennis_level: str = ""):
    """Update existing player with name and tennis level collected during coaching"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_id}"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Prepare update data
        update_data = {"fields": {}}
        if name:
            update_data["fields"]["name"] = name
        if tennis_level:
            update_data["fields"]["tennis_level"] = tennis_level
        
        response = requests.patch(url, headers=headers, json=update_data)
        
        return response.status_code == 200
    except Exception as e:
        return False

def create_new_player(email: str, name: str = "", tennis_level: str = ""):
    try:
        # Normalize email to lowercase
        email = email.lower().strip()
        
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Use provided name, or extract from email, or leave empty for Coach TA collection
        if name:
            player_name = name
        else:
            # For new players, leave empty - Coach TA will collect it
            player_name = ""
        
        # Prepare fields
        fields = {
            "email": email,  # Now lowercase
            "name": player_name,
            "primary_goals": [],
            "personality_notes": "",
            "total_sessions": 1,
            "first_session_date": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "player_status": "Active"
        }
        
        # Only add tennis_level if it has a valid value
        if tennis_level and tennis_level in ["Beginner", "Intermediate", "Advanced"]:
            fields["tennis_level"] = tennis_level
        # Don't include tennis_level field at all if empty - let Airtable handle default
        
        data = {"fields": fields}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def detect_session_end(message_content: str, conversation_history: list = None) -> dict:
    """
    Intelligent session end detection with context awareness
    Returns: {'should_end': bool, 'confidence': str, 'needs_confirmation': bool}
    """
    message_lower = message_content.lower().strip()
    
    # DEFINITIVE ending phrases (high confidence)
    definitive_endings = [
        "end session", "stop session", "finish session", "done for today",
        "that's all for today", "see you next time", "until next time",
        "goodbye coach", "bye coach", "thanks coach, bye", "session over"
    ]
    
    # LIKELY ending phrases (need confirmation)
    likely_endings = [
        "thanks coach", "thank you coach", "great session", "good session",
        "that's helpful", "i'll practice that", "got it, thanks", "perfect, thanks"
    ]
    
    # AMBIGUOUS words (only if conversation is winding down)
    ambiguous_endings = ["thanks", "thank you", "bye", "done", "great", "perfect"]
    
    # Check definitive endings
    for phrase in definitive_endings:
        if phrase in message_lower:
            return {'should_end': True, 'confidence': 'high', 'needs_confirmation': False}
    
    # Check likely endings (coaching-specific)
    for phrase in likely_endings:
        if phrase in message_lower:
            return {'should_end': True, 'confidence': 'medium', 'needs_confirmation': True}
    
    # Check ambiguous endings with context
    if any(word in message_lower for word in ambiguous_endings):
        # Only trigger if message is short AND seems conclusive
        word_count = len(message_lower.split())
        if word_count <= 3:
            # Check conversation context for winding down signals
            if conversation_history and len(conversation_history) >= 4:
                recent_messages = [msg['content'].lower() for msg in conversation_history[-4:] if msg['role'] == 'user']
                
                # Look for patterns suggesting session is ending
                coaching_complete_signals = [
                    "got it", "understand", "will practice", "makes sense", 
                    "clear", "helpful", "that helps", "i see"
                ]
                
                has_completion_signals = any(
                    any(signal in msg for signal in coaching_complete_signals) 
                    for msg in recent_messages
                )
                
                if has_completion_signals:
                    return {'should_end': True, 'confidence': 'low', 'needs_confirmation': True}
    
    # NOT an ending
    return {'should_end': False, 'confidence': 'none', 'needs_confirmation': False}

def generate_session_end_confirmation(user_message: str, confidence: str) -> str:
    """Generate appropriate confirmation message based on confidence level"""
    
    if confidence == 'medium':
        return ("Sounds like you're ready to wrap up! Should we end today's session? "
                "I'll save everything we covered and you can always come back for more coaching. "
                "Just say 'yes' to finish or keep asking questions! 🎾")
    
    elif confidence == 'low':
        return ("Are we finishing up for today? If you'd like to end the session, just say 'yes' "
                "and I'll save our progress. Or feel free to ask me anything else! 🎾")
    
    else:
        return ("Ready to finish today's coaching? Say 'yes' to end the session or "
                "keep the conversation going! 🎾")

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
    Get recent summaries for a specific player - ORIGINAL WITH PLAYER FILTERING
    """
    try:
        # First, get the player's email to match summaries
        player_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        player_response = requests.get(player_url, headers=headers)
        if player_response.status_code != 200:
            return []
            
        player_email = player_response.json().get('fields', {}).get('email', '')
        
        # Get all summaries and find ones for this email
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Session_Summaries"
        params = {
            "sort[0][field]": "session_number", 
            "sort[0][direction]": "desc",
            "maxRecords": 50  # Get more to search through
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            all_records = response.json().get('records', [])
            
            # Match summaries by checking if player_id links to our player
            matching_summaries = []
            for record in all_records:
                fields = record.get('fields', {})
                
                # NEW: Actually check if this summary belongs to our player
                player_ids = fields.get('player_id', [])
                if isinstance(player_ids, list) and player_record_id in player_ids:
                    # Skip if no technical_focus (empty summary)
                    if not fields.get('technical_focus'):
                        continue
                        
                    matching_summaries.append({
                        'session_number': fields.get('session_number', 0),
                        'technical_focus': fields.get('technical_focus', ''),
                        'homework_assigned': fields.get('homework_assigned', ''),
                        'next_session_focus': fields.get('next_session_focus', ''),
                        'key_breakthroughs': fields.get('key_breakthroughs', ''),
                        'condensed_summary': fields.get('condensed_summary', '')
                    })
            
            return matching_summaries[:limit]
        return []
    except Exception as e:
        return []

# ENHANCED: Welcome message generation with better context
def generate_personalized_welcome_message(player_name: str, session_number: int, recent_summaries: list, is_returning: bool) -> str:
    """
    Generate a concise, scannable welcome message for returning players
    """
    if not is_returning or not recent_summaries:
        # NEW PLAYER - Coach TA introduction sequence
        return "Hi! I'm Coach TA, your personal tennis coach. What's your name?"
    
    # RETURNING PLAYER with history
    last_session = recent_summaries[0]
    
    # Build concise welcome
    welcome_parts = [f"Hi {player_name}! Coach TA here. **Session #{session_number}** 🎾"]
    
    # Add ONE key item from last session (priority order)
    if last_session.get('technical_focus'):
        focus = last_session['technical_focus'][:50] + "..." if len(last_session['technical_focus']) > 50 else last_session['technical_focus']
        welcome_parts.append(f"\n**Last time:** {focus}")
    
    # Add homework check if exists
    if last_session.get('homework_assigned'):
        homework = last_session['homework_assigned'][:60] + "..." if len(last_session['homework_assigned']) > 60 else last_session['homework_assigned']
        welcome_parts.append(f"\n**Your practice:** {homework}")
        welcome_parts.append(f"\n\n*How did it go?*")
    else:
        welcome_parts.append(f"\n\nWhat would you like to work on today?")
    
    return "".join(welcome_parts)

# ENHANCED: Build conversational prompt with coaching history
def build_conversational_prompt_with_history(user_question: str, context_chunks: list, conversation_history: list, coaching_history: list = None, player_name: str = None, player_level: str = None) -> str:
    """Build Claude prompt with proper player context and memory"""
    
    # Check if this is intro
    is_intro = not st.session_state.get("intro_completed", True)
    
    if is_intro:
        # NEW PLAYER INTRODUCTION PROMPT
        intro_prompt = """You are Coach TA. Be natural and conversational.

INTRODUCTION FLOW:
- Start: "Hi! I'm Coach TA, your personal tennis coach. What's your name?"
- After name: "Nice to meet you, [Name]! I am excited, tell me about your tennis. You been playing long?"
- After experience: "What's challenging you most on court right now?"
- Then transition: "Great! How about we work on [specific area] today?"

Keep responses SHORT (1-2 sentences max). Be enthusiastic but concise."""
        
        # Add current conversation context for intro
        history_text = ""
        if conversation_history:
            history_text = "\nCurrent conversation:\n"
            for msg in conversation_history[-6:]:  # Last 6 exchanges
                role = "Player" if msg['role'] == 'user' else "Coach TA"
                history_text += f"{role}: {msg['content']}\n"
        
        context_text = "\n\n".join([chunk.get('text', '') for chunk in context_chunks if chunk.get('text')])
        
        return f"""{intro_prompt}
{history_text}

Tennis Knowledge: {context_text}

Player says: "{user_question}"

Respond naturally as Coach TA:"""
    
    else:
        # REGULAR COACHING PROMPT WITH FULL CONTEXT
        player_context = ""
        if player_name and player_level:
            player_context = f"Player: {player_name} (Level: {player_level})\n"
        
        coaching_prompt = f"""You are Coach TA coaching {player_name or 'the player'}.
{player_context}

You provide direct, actionable tennis coaching advice. 

COACHING APPROACH:
- Ask 1-2 quick questions about their specific situation
- Give ONE specific tip or drill appropriate for {player_level or 'their current'} level  
- End with encouragement like "How about we try this?" or "Sound good?"
- Keep responses SHORT (2-3 sentences total)
- Be encouraging and practical
- Focus on actionable advice they can practice alone

MEMORY RULES:
- NEVER ask about their level - you know they are {player_level or 'at their current level'}
- NEVER ask their name - you are coaching {player_name or 'this player'}
- Remember what you suggested earlier in this session

NEVER say "Hi there" or greet again - you're already in conversation.
NEVER include meta-commentary about your process.
Just give direct coaching advice."""
        
        # Add previous session context
        history_text = ""
        if coaching_history and len(coaching_history) > 0:
            last_session = coaching_history[0]
            if last_session.get('technical_focus'):
                history_text += f"\nPrevious session focus: {last_session['technical_focus']}"
        
        # Add current conversation context
        if conversation_history and len(conversation_history) > 1:
            history_text += "\nCurrent session conversation:\n"
            for msg in conversation_history[-10:]:  # Last 10 exchanges to maintain context
                role = "Player" if msg['role'] == 'user' else "Coach TA"
                history_text += f"{role}: {msg['content']}\n"
        
        context_text = "\n\n".join([chunk.get('text', '') for chunk in context_chunks if chunk.get('text')])
        
        return f"""{coaching_prompt}
{history_text}

Tennis Knowledge: {context_text}

Player says: "{user_question}"

Give direct coaching advice:"""

def extract_name_from_response(user_message: str) -> str:
    """
    Extract player name from their response to "What's your name?"
    """
    # Simple extraction - look for common patterns
    message = user_message.strip()
    
    # Handle common responses
    if message.lower().startswith(("i'm ", "im ", "i am ")):
        return message.split(" ", 2)[2] if len(message.split()) > 2 else message.split(" ", 1)[1]
    elif message.lower().startswith(("my name is ", "name is ")):
        return message.split("is ", 1)[1]
    elif message.lower().startswith(("call me ", "it's ", "its ")):
        return message.split(" ", 1)[1]
    else:
        # Assume the whole message is the name (most common case)
        return message.title()

def assess_player_level_from_conversation(conversation_history: list, claude_client) -> str:
    # Extract player responses
    player_responses = []
    for msg in conversation_history:
        if msg["role"] == "user":
            player_responses.append(msg["content"])
    
    if len(player_responses) < 2:
        return "Beginner"
    
    # Combine all tennis-related responses
    all_responses = " ".join(player_responses[1:]).lower()
    
    # Smart logic-based assessment
    advanced_indicators = ["kick serve", "slice serve", "topspin", "years", "competitive", "tournament", "league", "advanced", "intermediate"]
    intermediate_indicators = ["year", "months", "consistent", "working on", "improving", "regular"]
    
    # Check for advanced indicators
    if any(indicator in all_responses for indicator in advanced_indicators):
        return "Advanced"
    
    # Check for intermediate indicators  
    if any(indicator in all_responses for indicator in intermediate_indicators):
        return "Intermediate"
    
    # Default to beginner
    return "Beginner"

def handle_introduction_sequence(user_message: str, claude_client):
    """
    Handle the introduction sequence for new players with invisible level assessment
    """
    intro_state = st.session_state.get("intro_state", "waiting_for_name")
    
    if intro_state == "waiting_for_name":
        # Extract name from user response
        player_name = extract_name_from_response(user_message)
        if player_name:
            st.session_state.collected_name = player_name
            st.session_state.intro_state = "collecting_experience"
            return f"Nice to meet you, {player_name}! I am excited, tell me about your tennis. You been playing long?"    
    
    elif intro_state == "collecting_experience":
        st.session_state.intro_state = "ready_for_assessment"
        return "What's your biggest challenge on court right now? What shots feel most comfortable to you?"
    
    elif intro_state == "ready_for_assessment":
        # Now we have enough conversation to assess level
        assessed_level = assess_player_level_from_conversation(st.session_state.messages, claude_client)
        
        # Update player record with collected name and assessed level
        success = update_player_info(
            st.session_state.player_record_id,
            st.session_state.collected_name,
            assessed_level
        )
        
        if success:
            st.session_state.intro_completed = True
            st.session_state.intro_state = "complete"
            return "Great! What would you like to work on today?"
        else:
            # Fallback - continue anyway
            st.session_state.intro_completed = True
            return "Perfect! What would you like to work on today?"
    
    return None

def setup_player_session_with_continuity(player_email: str):
    """
    Enhanced player setup with proper continuity system - WITH COACH TA INTRO
    """
    existing_player = find_player_by_email(player_email)
    
    if existing_player:
        # Returning player
        player_data = existing_player['fields']
        st.session_state.player_record_id = existing_player['id']
        st.session_state.is_returning_player = True
        player_name = player_data.get('name', 'there')
        session_number = player_data.get('total_sessions', 0) + 1
        
        with st.spinner("Loading your coaching history..."):
            recent_summaries = get_player_recent_summaries(existing_player['id'], 2)
            st.session_state.coaching_history = recent_summaries
        
        # Use the updated concise welcome message function
        welcome_msg = generate_personalized_welcome_message(
            player_name, 
            session_number, 
            recent_summaries, 
            True  # is_returning = True
        )
        
        update_player_session_count(existing_player['id'])
        
    else:
        # NEW PLAYER
        new_player = create_new_player(player_email, "", "")  # Empty name and level initially
        
        if new_player:
            st.session_state.player_record_id = new_player['id']
            st.session_state.is_returning_player = False
            st.session_state.coaching_history = []
            
            # Set introduction state
            st.session_state.intro_state = "waiting_for_name"
            st.session_state.intro_completed = False
            
            welcome_msg = "Hi! I'm Coach TA, your personal tennis coach. What's your name?"
        else:
            st.error("Error creating player profile. Please try again.")
            return None
    
    return welcome_msg

def get_current_player_info(player_record_id: str) -> tuple:
    """Retrieve current player name and level from database"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            fields = response.json().get('fields', {})
            name = fields.get('name', '')
            level = fields.get('tennis_level', '')
            return name, level
        return '', ''
    except Exception as e:
        return '', ''

def is_valid_email(email: str) -> bool:
    """
    Robust email validation using regex pattern
    Future-proof and handles international domains
    """
    import re
    
    # Comprehensive email regex pattern
    # Handles: username+tags@subdomain.domain.extension
    pattern = r'''
        ^                       # Start of string
        [a-zA-Z0-9]             # First character must be alphanumeric
        [a-zA-Z0-9._%+-]*       # Username can contain letters, numbers, dots, underscores, percent, plus, hyphen
        [a-zA-Z0-9]             # Last character of username must be alphanumeric
        @                       # Required @ symbol
        [a-zA-Z0-9]             # Domain must start with alphanumeric
        [a-zA-Z0-9.-]*          # Domain can contain letters, numbers, dots, hyphens
        [a-zA-Z0-9]             # Domain must end with alphanumeric
        \.                      # Required dot before extension
        [a-zA-Z]{2,}            # Extension must be at least 2 letters
        $                       # End of string
    '''
    
    return re.match(pattern, email.strip(), re.VERBOSE) is not None

def generate_dynamic_session_ending(conversation_history: list, player_name: str = "") -> str:
    """
    Generate personalized, varied session ending messages focused on effort, learning, and motivation
    """
    import random
    
    # Analyze the session to personalize the message
    session_content = " ".join([msg['content'].lower() for msg in conversation_history if msg['role'] == 'user'])
    
    # Detect what they worked on
    techniques = []
    if any(word in session_content for word in ['forehand', 'forehand']):
        techniques.append('forehand')
    if any(word in session_content for word in ['backhand', 'backhand']):
        techniques.append('backhand')
    if any(word in session_content for word in ['serve', 'serving']):
        techniques.append('serve')
    if any(word in session_content for word in ['volley', 'net']):
        techniques.append('volleys')
    if any(word in session_content for word in ['footwork', 'movement']):
        techniques.append('footwork')
    
    # Effort acknowledgments (varied)
    effort_phrases = [
        f"Love your commitment today{f', {player_name}' if player_name else ''}!",
        "You really focused on the details today - that's how improvement happens!",
        f"Great questions today{f', {player_name}' if player_name else ''} - shows you're thinking like a player!",
        "I can see you're putting in the mental work - that's just as important as physical practice!",
        "Your dedication to getting better really shows!"
    ]
    
    # Learning/challenge acknowledgments
    if techniques:
        technique_work = techniques[0] if len(techniques) == 1 else f"{techniques[0]} and {techniques[1]}"
        learning_phrases = [
            f"Working on {technique_work} takes patience - you're on the right track!",
            f"Those {technique_work} adjustments we discussed will click with practice!",
            f"Remember, mastering {technique_work} is a process - every rep counts!",
            f"The {technique_work} work we covered today will pay off on court!"
        ]
    else:
        learning_phrases = [
            "The concepts we covered today will make more sense as you practice them!",
            "Breaking down technique like this is how real improvement happens!",
            "Those adjustments take time to feel natural - trust the process!",
            "Every detail we discussed today builds toward better tennis!"
        ]
    
    # Motivational closings
    motivation_phrases = [
        "Keep that curiosity and drive - it's your biggest asset! 🎾",
        "You've got the right mindset to take your game to the next level! 🎾",
        "Stay patient with yourself and trust the process - you're improving! 🎾",
        "That focus you showed today is what separates good players from great ones! 🎾",
        "Keep asking great questions and putting in the work - exciting progress ahead! 🎾"
    ]
    
    # Combine randomly
    effort = random.choice(effort_phrases)
    learning = random.choice(learning_phrases)
    motivation = random.choice(motivation_phrases)
    
    return f"{effort} {learning} {motivation}"

# ============== ADMIN INTERFACE FUNCTIONS ==============

def get_all_coaching_sessions():
    """Fetch all coaching sessions with player info for admin dropdown"""
    try:
        # Get all Active_Sessions
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {
            "sort[0][field]": "timestamp",
            "sort[0][direction]": "desc",
            "maxRecords": 500
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return []
        
        records = response.json().get('records', [])
        
        # Group by session_id and get session info
        sessions = {}
        for record in records:
            fields = record.get('fields', {})
            session_id = fields.get('session_id')
            player_ids = fields.get('player_id', [])
            
            if session_id and player_ids:
                player_id = player_ids[0] if isinstance(player_ids, list) else player_ids
                
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session_id': session_id,
                        'player_id': player_id,
                        'message_count': 0,
                        'first_message_time': fields.get('timestamp', ''),
                        'status': fields.get('session_status', 'unknown')
                    }
                
                sessions[session_id]['message_count'] += 1
        
        # Get player names
        session_list = []
        for session_data in sessions.values():
            player_name = get_player_name(session_data['player_id'])
            session_list.append({
                'session_id': session_data['session_id'],
                'player_name': player_name,
                'message_count': session_data['message_count'],
                'timestamp': session_data['first_message_time'],
                'status': session_data['status']
            })
        
        return sorted(session_list, key=lambda x: x['timestamp'], reverse=True)
        
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
        return []

def get_player_name(player_id: str) -> str:
    """Get player name by ID"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_id}"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            fields = response.json().get('fields', {})
            name = fields.get('name', 'Unknown Player')
            email = fields.get('email', '')
            return f"{name}" if name != 'Unknown Player' else email.split('@')[0] if email else 'Unknown'
        return 'Unknown Player'
    except Exception:
        return 'Unknown Player'

def get_conversation_messages(session_id: int) -> list:
    """Fetch all messages for a specific session"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {
            "filterByFormula": f"{{session_id}} = {session_id}",
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
                    'order': fields.get('message_order', 0),
                    'timestamp': fields.get('timestamp', ''),
                    'coaching_resources': fields.get('coaching_resources_used', 0),
                    'resource_details': fields.get('resource_details', '')
                })
            
            return messages
        return []
    except Exception as e:
        st.error(f"Error fetching conversation: {e}")
        return []

def display_admin_interface():
    """Main admin interface display"""
    st.title("🔧 Tennis Coach AI - Admin Interface")
    st.markdown("### Session Management & Analytics")
    st.markdown("---")
    
    # Session selector
    sessions = get_all_coaching_sessions()
    
    if not sessions:
        st.warning("No coaching sessions found in the database.")
        return
    
    # Create session options for dropdown
    session_options = {}
    for session in sessions:
        timestamp = session['timestamp']
        try:
            # Parse timestamp and format it nicely
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%m/%d %H:%M")
        except:
            formatted_time = "Unknown time"
        
        status_emoji = "✅" if session['status'] == 'completed' else "🟡"
        display_name = f"{status_emoji} {session['player_name']} - {session['message_count']} msgs - {formatted_time}"
        session_options[display_name] = session['session_id']
    
    # Session selection dropdown
    selected_session_display = st.selectbox(
        "🎾 Select Coaching Session",
        options=list(session_options.keys()),
        help="Choose a session to view the full conversation and analytics"
    )
    
    if selected_session_display:
        selected_session_id = session_options[selected_session_display]
        
        # Display session details
        session_info = next(s for s in sessions if s['session_id'] == selected_session_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Player", session_info['player_name'])
        with col2:
            st.metric("Messages", session_info['message_count'])
        with col3:
            status_display = "Completed ✅" if session_info['status'] == 'completed' else "Active 🟡"
            st.metric("Status", status_display)
        
        st.markdown("---")
        
        # Fetch and display conversation
        messages = get_conversation_messages(selected_session_id)
        
        if messages:
            st.markdown("### 💬 Conversation Log")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["Chat View", "Analytics"])
            
            with tab1:
                # Display chat-style conversation
                for msg in messages:
                    role = msg['role']
                    content = msg['content']
                    resources_used = msg.get('coaching_resources', 0)
                    
                    if role == 'player':
                        # Player message - left aligned, blue background
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                            <div style="background-color: #E3F2FD; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #BBDEFB;">
                                <strong>🧑‍🎓 Player:</strong><br>
                                {content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:  # coach
                        # Coach message - right aligned, green background
                        resources_indicator = f" 📚 {resources_used} resources" if resources_used > 0 else ""
                        st.markdown(f"""
                        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                            <div style="background-color: #E8F5E8; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #C8E6C9;">
                                <strong>🎾 Coach TA:</strong>{resources_indicator}<br>
                                {content}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show resource details if available
                        if resources_used > 0 and msg.get('resource_details'):
                            with st.expander(f"📊 View {resources_used} coaching resources used"):
                                st.text(msg['resource_details'])
            
            with tab2:
                # Analytics view
                st.markdown("### 📊 Session Analytics")
                
                # Calculate analytics
                total_messages = len(messages)
                player_messages = len([m for m in messages if m['role'] == 'player'])
                coach_messages = len([m for m in messages if m['role'] == 'coach'])
                total_resources = sum(m.get('coaching_resources', 0) for m in messages)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Messages", total_messages)
                with col2:
                    st.metric("Player Messages", player_messages)
                with col3:
                    st.metric("Coach Responses", coach_messages)
                with col4:
                    st.metric("Resources Used", total_resources)
                
                # Resource usage breakdown
                if total_resources > 0:
                    st.markdown("#### 📚 Resource Usage by Response")
                    resource_data = []
                    for i, msg in enumerate(messages):
                        if msg['role'] == 'coach' and msg.get('coaching_resources', 0) > 0:
                            resource_data.append({
                                'Response #': i + 1,
                                'Resources Used': msg['coaching_resources'],
                                'Preview': msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                            })
                    
                    if resource_data:
                        df = pd.DataFrame(resource_data)
                        st.dataframe(df, use_container_width=True)
                
                # Message length analysis
                st.markdown("#### 📏 Message Length Analysis")
                player_lengths = [len(m['content'].split()) for m in messages if m['role'] == 'player']
                coach_lengths = [len(m['content'].split()) for m in messages if m['role'] == 'coach']
                
                if player_lengths and coach_lengths:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Avg Player Message", f"{sum(player_lengths)//len(player_lengths)} words")
                    with col2:
                        st.metric("Avg Coach Response", f"{sum(coach_lengths)//len(coach_lengths)} words")
        
        else:
            st.warning("No messages found for this session.")
    
    # Exit admin mode
    st.markdown("---")
    if st.button("🏃‍♂️ Exit Admin Mode", type="primary"):
        st.session_state.admin_mode = False
        st.rerun()

def main():
    st.set_page_config(
        page_title="Tennis Coach AI",
        page_icon="🎾",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # ============== ADMIN MODE CHECK ==============
    # Check for admin mode trigger - TEMPORARILY DISABLED
    if False:  # Changed from st.session_state.get('admin_mode', False):
        display_admin_interface()
        return
    
    # ============== EXISTING CODE CONTINUES ==============
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
            
            # REMOVED: player_name input field - Coach TA will collect this
            
            if st.form_submit_button("Start Coaching Session", type="primary"):
                if not player_email:
                    st.error("Please enter your email address.")
                elif not is_valid_email(player_email):
                    st.error("⚠️ Please enter a valid email address (example: yourname@domain.com)")
                else:
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
        # ============== ADMIN TRIGGER CHECK - NEW ADDITION ==============
        if prompt.strip().lower() == "hilly spike":
            st.write("Admin trigger detected - but admin mode is disabled for testing")
            return
        
        # ============== EXISTING CHAT PROCESSING CONTINUES ==============
        # Smart session end detection
        end_result = detect_session_end(prompt, st.session_state.messages)
        
        if end_result['should_end']:
            if end_result['needs_confirmation']:
                # Set confirmation state instead of ending immediately
                st.session_state.pending_session_end = True
                st.session_state.end_confidence = end_result['confidence']
            else:
                # High confidence - end immediately
                st.session_state.session_ending = True
        
        # Handle confirmation responses
        if st.session_state.get("pending_session_end") and prompt.lower().strip() in ["yes", "y", "yeah", "yep", "sure"]:
            st.session_state.session_ending = True
            st.session_state.pending_session_end = False
        elif st.session_state.get("pending_session_end") and prompt.lower().strip() in ["no", "n", "nope", "not yet", "continue"]:
            st.session_state.pending_session_end = False
        
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
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # NEW: Handle introduction sequence for new players
        if not st.session_state.get("intro_completed", True):  # True for returning players
            intro_response = handle_introduction_sequence(prompt, claude_client)
            if intro_response:
                with st.chat_message("assistant"):
                    st.markdown(intro_response)
                
                st.session_state.message_counter += 1
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": intro_response
                })
                
                # Log to SSS Active_Sessions
                if st.session_state.get("player_record_id"):
                    log_message_to_sss(
                        st.session_state.player_record_id,
                        st.session_state.session_id,
                        st.session_state.message_counter,
                        "assistant",
                        intro_response
                    )
                return  # Don't process as normal coaching message yet
        
        # Handle session end confirmation
        if st.session_state.get("pending_session_end"):
            confidence = st.session_state.get("end_confidence", "medium")
            confirmation_msg = generate_session_end_confirmation(prompt, confidence)
            
            with st.chat_message("assistant"):
                st.markdown(confirmation_msg)
            
            st.session_state.message_counter += 1
            st.session_state.messages.append({
                "role": "assistant", 
                "content": confirmation_msg
            })
            
            # Log confirmation message
            if st.session_state.get("player_record_id"):
                log_message_to_sss(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    confirmation_msg
                )
            return
        
        # If session is ending, provide closing response and mark as completed
        if st.session_state.get("session_ending"):
            with st.chat_message("assistant"):
                # Get player name for personalized ending message
                player_name, _ = get_current_player_info(st.session_state.get("player_record_id", ""))
                closing_response = generate_dynamic_session_ending(st.session_state.messages, player_name)
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
                    session_marked = mark_session_completed(
                        st.session_state.player_record_id,
                        st.session_state.session_id
                    )
                    if session_marked:
                        st.success("✅ Session marked as completed!")
                        
                        # Generate session summary
                        with st.spinner("🧠 Generating session summary..."):
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
                    coaching_history = st.session_state.get('coaching_history', [])
                    
                    # Get current player info from database
                    player_name, player_level = get_current_player_info(st.session_state.get("player_record_id", ""))
                    
                    full_prompt = build_conversational_prompt_with_history(
                        prompt, 
                        chunks, 
                        st.session_state.messages[:-1],
                        coaching_history,
                        player_name,
                        player_level
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
