import streamlit as st
import os
import json
from typing import List, Dict
import time
import pandas as pd          # NEW
from datetime import datetime # NEW

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
        
        # Process resource details if chunks provided
        resource_count = 0
        resource_details = ""
        
        if chunks and role == "assistant":
            resource_count = len(chunks)
            resource_details_list = []
            for i, chunk in enumerate(chunks):
                relevance_score = round(chunk.get('score', 0), 3)
                source = chunk.get('source', 'Unknown')
                topics = chunk.get('topics', 'General')
                resource_details_list.append(
                    f"Resource {i+1}: {relevance_score} relevance | {topics} | {source}"
                )
            resource_details = "\n".join(resource_details_list)
        
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
                "session_status": "active",
                "coaching_resources_used": resource_count,
                "resource_details": resource_details[:100000] if resource_details else ""
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 200
        
    except Exception as e:
        return False

def log_message_to_conversation_log(player_record_id: str, session_id: str, message_order: int, 
                                   role: str, content: str, chunks=None) -> bool:
    """Enhanced logging that includes resource relevance data to Conversation_Log table"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Conversation_Log"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        # Process resource details if chunks provided
        resource_count = 0
        resource_details = ""
        
        if chunks and role == "assistant":
            resource_count = len(chunks)
            resource_details_list = []
            for i, chunk in enumerate(chunks):
                relevance_score = round(chunk.get('score', 0), 3)
                source = chunk.get('source', 'Unknown')
                topics = chunk.get('topics', 'General')
                resource_details_list.append(
                    f"Resource {i+1}: {relevance_score} relevance | {topics} | {source}"
                )
            resource_details = "\n".join(resource_details_list)
        
        # Get the session record ID to link to
        # First, find the Active_Sessions record with this session_id
        session_search_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        session_id_number = int(''.join(filter(str.isdigit, session_id))) if session_id else 1
        
        search_params = {
            "filterByFormula": f"{{session_id}} = {session_id_number}",
            "maxRecords": 1
        }
        
        session_response = requests.get(session_search_url, headers=headers, params=search_params)
        session_record_id = None
        
        if session_response.status_code == 200:
            session_records = session_response.json().get('records', [])
            if session_records:
                session_record_id = session_records[0]['id']
        
        # Prepare data for Conversation_Log
        data = {
            "fields": {
                "message_order": message_order,
                "role": "coach" if role == "assistant" else "player",
                "message_content": content[:100000],
                "coaching_resources_used": resource_count,
                "resource_details": resource_details[:100000] if resource_details else ""
            }
        }
        
        # Add session_id link if we found the session record
        if session_record_id:
            data["fields"]["session_id"] = [session_record_id]
        
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

import re
from datetime import datetime, timedelta

def calculate_days_since_last_session(player_record_id: str) -> int:
    """
    Calculate days since last session by looking at Active_Sessions timestamps
    """
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {
            "sort[0][field]": "timestamp",
            "sort[0][direction]": "desc",
            "maxRecords": 50
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            
            # Find most recent session for this player
            for record in records:
                fields = record.get('fields', {})
                record_player_ids = fields.get('player_id', [])
                
                if isinstance(record_player_ids, list) and player_record_id in record_player_ids:
                    last_timestamp = fields.get('timestamp', '')
                    if last_timestamp:
                        try:
                            # Parse the timestamp
                            last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            now_dt = datetime.now(last_dt.tzinfo)
                            days_diff = (now_dt - last_dt).days
                            return days_diff
                        except:
                            pass
                    break
        
        return 7  # Default to 1 week if can't determine
    except Exception as e:
        return 7  # Default fallback


def analyze_session_tone(session_summary: dict) -> str:
    """
    Analyze the tone/mood of the last session from summary content
    """
    if not session_summary:
        return "neutral"
    
    # Combine all text fields for analysis
    all_text = " ".join([
        session_summary.get('technical_focus', ''),
        session_summary.get('key_breakthroughs', ''),
        session_summary.get('mental_game_notes', ''),
        session_summary.get('condensed_summary', '')
    ]).lower()
    
    if not all_text.strip():
        return "neutral"
    
    # Define tone indicators
    positive_indicators = [
        'breakthrough', 'progress', 'improvement', 'great', 'excellent', 'good',
        'clicked', 'got it', 'makes sense', 'comfortable', 'confident',
        'working well', 'success', 'better', 'improved', 'solid'
    ]
    
    challenging_indicators = [
        'struggle', 'difficult', 'frustrating', 'hard time', 'trouble',
        'inconsistent', 'issues', 'problems', 'challenging', 'tough',
        'need work', 'focus on', 'fix', 'work on'
    ]
    
    technical_indicators = [
        'grip', 'stance', 'follow-through', 'technique', 'mechanics',
        'form', 'adjustment', 'forehand', 'backhand', 'serve', 'volley',
        'footwork', 'swing', 'contact', 'timing'
    ]
    
    mental_indicators = [
        'confidence', 'nervous', 'pressure', 'focus', 'mindset',
        'mental', 'thinking', 'calm', 'anxiety', 'concentration'
    ]
    
    # Count indicators
    positive_count = sum(1 for indicator in positive_indicators if indicator in all_text)
    challenging_count = sum(1 for indicator in challenging_indicators if indicator in all_text)
    technical_count = sum(1 for indicator in technical_indicators if indicator in all_text)
    mental_count = sum(1 for indicator in mental_indicators if indicator in all_text)
    
    # Determine primary tone
    if positive_count >= 2 and positive_count > challenging_count:
        return "positive"
    elif challenging_count >= 2 and challenging_count > positive_count:
        return "challenging"
    elif technical_count >= 2:
        return "technical"
    elif mental_count >= 2:
        return "mental"
    else:
        return "neutral"


def determine_frequency_pattern(total_sessions: int, days_since_last: int) -> str:
    """
    Determine player's engagement pattern
    """
    if total_sessions <= 2:
        return "new_returner"
    elif total_sessions >= 8:
        return "committed"
    elif days_since_last <= 7:
        return "regular"
    elif days_since_last <= 14:
        return "occasional"
    else:
        return "infrequent"


def get_greeting_variation(category: str, used_greetings: list) -> str:
    """
    Get a greeting variation that hasn't been used recently
    """
    greeting_bank = {
        "same_day": [
            "Back already! How's it going?",
            "Twice in one day - I love the dedication!",
            "Ready for round two?"
        ],
        "next_day": [
            "Back for more! How are you feeling?",
            "Love the commitment - ready to keep working?",
            "Day two! How's everything feeling?"
        ],
        "positive_recent": [
            "Hey {name}! Still feeling good about that progress?",
            "{name}! How's that confidence been?",
            "Hi {name}! I bet you've been thinking about that breakthrough!"
        ],
        "challenging_return": [
            "Hey {name}! How are you feeling today?",
            "{name}! Ready to tackle some tennis?",
            "Hi {name}! How's everything been going?"
        ],
        "technical_followup": [
            "Hey {name}! How's that technique been working out?",
            "{name}! Have you been practicing what we worked on?",
            "Hi {name}! How's that adjustment feeling?"
        ],
        "mental_checkin": [
            "Hey {name}! How's the confidence been?",
            "{name}! How's your mindset been on court?",
            "Hi {name}! How are you feeling about your game?"
        ],
        "been_a_while": [
            "{name}! Great to have you back!",
            "Hey {name}! It's been a while - how have you been?",
            "{name}! Good to see you again!"
        ],
        "long_time": [
            "{name}! Wow, it's been a while - how have you been?",
            "Hey {name}! Great to see you back after so long!",
            "{name}! Good to have you back - how's life been?"
        ],
        "regular_casual": [
            "Hey {name}! How's it going?",
            "{name}! Good to see you again!",
            "Hi {name}! How have you been?"
        ],
        "committed_encouraging": [
            "Hey {name}! Love seeing you back so consistently!",
            "{name}! Your dedication is impressive - how are you feeling?",
            "Hi {name}! Ready for another great session?"
        ],
        "new_returner_welcoming": [
            "Hey {name}! Good to see you back!",
            "{name}! Nice to see you're staying with it!",
            "Hi {name}! How has tennis been treating you?"
        ]
    }
    
    variations = greeting_bank.get(category, greeting_bank["regular_casual"])
    
    # Filter out recently used greetings
    available = [g for g in variations if g not in used_greetings]
    
    # If all were used recently, use the oldest one
    if not available:
        available = variations
    
    # Return the first available variation
    return available[0]


def generate_smart_greeting(player_name: str, player_record_id: str, last_session_summary: dict, total_sessions: int) -> str:
    """
    Generate intelligent, varied greeting based on player history and context
    """
    # Get session memory for recent greetings
    used_greetings = st.session_state.get('recent_greetings', [])
    
    # Calculate context factors
    days_since = calculate_days_since_last_session(player_record_id)
    session_tone = analyze_session_tone(last_session_summary)
    frequency_pattern = determine_frequency_pattern(total_sessions, days_since)
    
    # Decision tree logic
    greeting_category = "regular_casual"  # default
    
    # Time-based priorities (override others)
    if days_since == 0:
        greeting_category = "same_day"
    elif days_since == 1:
        greeting_category = "next_day"
    elif days_since >= 22:
        greeting_category = "long_time"
    elif days_since >= 11:
        greeting_category = "been_a_while"
    else:
        # Context-based greetings for recent visits
        if session_tone == "positive":
            greeting_category = "positive_recent"
        elif session_tone == "challenging":
            greeting_category = "challenging_return"
        elif session_tone == "technical":
            greeting_category = "technical_followup"
        elif session_tone == "mental":
            greeting_category = "mental_checkin"
        else:
            # Frequency-based defaults
            if frequency_pattern == "committed":
                greeting_category = "committed_encouraging"
            elif frequency_pattern == "new_returner":
                greeting_category = "new_returner_welcoming"
            else:
                greeting_category = "regular_casual"
    
    # Get the greeting variation
    greeting_template = get_greeting_variation(greeting_category, used_greetings)
    greeting = greeting_template.format(name=player_name)
    
    # Store in session memory (keep last 3)
    recent_greetings = st.session_state.get('recent_greetings', [])
    recent_greetings.append(greeting)
    st.session_state.recent_greetings = recent_greetings[-3:]  # Keep last 3
    
    return greeting


def generate_followup_message(player_name: str, last_session_summary: dict, session_tone: str) -> str:
    """
    Generate the second message based on what happened last session
    """
    if not last_session_summary:
        return "What's on your mind for today's session?"
    
    # Priority 1: Homework/practice check
    homework = last_session_summary.get('homework_assigned', '').strip()
    if homework:
        homework_preview = homework[:75] + "..." if len(homework) > 75 else homework
        return f"Did you get a chance to practice what we discussed? {homework_preview} How did it go?"
    
    # Priority 2: Breakthrough follow-up
    breakthroughs = last_session_summary.get('key_breakthroughs', '').strip()
    if breakthroughs and session_tone == "positive":
        breakthrough_preview = breakthroughs[:60] + "..." if len(breakthroughs) > 60 else breakthroughs
        return f"How has that breakthrough been working out? {breakthrough_preview}"
    
    # Priority 3: Technical follow-up
    technical_focus = last_session_summary.get('technical_focus', '').strip()
    if technical_focus:
        # Extract the main technique mentioned
        tech_words = ["forehand", "backhand", "serve", "volley", "grip", "stance", "footwork"]
        mentioned_tech = None
        for tech in tech_words:
            if tech in technical_focus.lower():
                mentioned_tech = tech
                break
        
        if mentioned_tech:
            return f"How has that {mentioned_tech} work been going since last time?"
        else:
            tech_preview = technical_focus[:50] + "..." if len(technical_focus) > 50 else technical_focus
            return f"How has the work on {tech_preview.lower()} been going?"
    
    # Priority 4: Next session focus
    next_focus = last_session_summary.get('next_session_focus', '').strip()
    if next_focus:
        focus_preview = next_focus[:60] + "..." if len(next_focus) > 60 else next_focus
        return f"Ready to work on what we planned? {focus_preview}"
    
    # Default fallback
    return "What would you like to focus on today?"


def enhanced_generate_personalized_welcome_message(player_name: str, session_number: int, recent_summaries: list, is_returning: bool) -> tuple:
    """
    Generate two-part welcome message: greeting + follow-up
    Returns: (greeting_message, followup_message, should_delay)
    """
    if not is_returning or not recent_summaries:
        # NEW PLAYER - single message
        return ("Hi! I'm Coach TA, your personal tennis coach. What's your name?", None, False)
    
    # RETURNING PLAYER - two message system
    last_session = recent_summaries[0] if recent_summaries else {}
    
    # Generate smart greeting (Message 1)
    player_record_id = st.session_state.get('player_record_id', '')
    total_sessions = session_number  # Approximate
    
    greeting = generate_smart_greeting(player_name, player_record_id, last_session, total_sessions)
    
    # Generate follow-up message (Message 2)
    session_tone = analyze_session_tone(last_session)
    followup = generate_followup_message(player_name, last_session, session_tone)
    
    return (greeting, followup, True)  # True = should delay second message

# ENHANCED: Welcome message generation with better context
def old_generate_personalized_welcome_message(player_name: str, session_number: int, recent_summaries: list, is_returning: bool) -> str:
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
    """
    Simple conversational assessment - when in doubt, default to Beginner
    """
    # Extract player responses only (skip the name collection)
    player_responses = []
    for msg in conversation_history:
        if msg["role"] == "user":
            player_responses.append(msg["content"])
    
    if len(player_responses) < 2:
        return "Beginner"  # Default for insufficient data
    
    # Combine all tennis-related responses (skip first which is usually name)
    all_responses = " ".join(player_responses[1:]).lower()
    
    # STEP 1: Check for explicit beginner indicators
    beginner_phrases = [
        "just started", "new to tennis", "beginner", "never played", 
        "first time", "starting out", "very new", "complete beginner"
    ]
    
    if any(phrase in all_responses for phrase in beginner_phrases):
        return "Beginner"
    
    # STEP 2: Look for time indicators  
    import re
    
    # Look for "less than" patterns that indicate beginner
    less_than_patterns = [
        r"less than.*year", r"under.*year", r"not even.*year",
        r"few months", r"couple.*months", r"\d+.*months"
    ]
    
    if any(re.search(pattern, all_responses) for pattern in less_than_patterns):
        return "Beginner"
    
    # Look for specific month mentions (if 6 months or less = beginner)
    month_numbers = re.findall(r'(\d+)\s*months?', all_responses)
    if month_numbers:
        max_months = max(int(month) for month in month_numbers)
        if max_months < 12:  # Less than a year
            return "Beginner"
    
    # STEP 3: Look for year indicators
    year_patterns = [
        r'(\d+)\s*years?', r'(\d+)\s*yrs?', 
        r'about\s*(\d+)\s*years?', r'around\s*(\d+)\s*years?',
        r'over\s*(\d+)\s*years?', r'more than\s*(\d+)\s*years?'
    ]
    
    years_mentioned = []
    for pattern in year_patterns:
        matches = re.findall(pattern, all_responses)
        years_mentioned.extend([int(match) for match in matches])
    
    # If less than 1 year mentioned, still beginner
    if years_mentioned and max(years_mentioned) < 1:
        return "Beginner"
    
    # STEP 4: If 1+ years mentioned, check frequency and lessons
    if years_mentioned and max(years_mentioned) >= 1:
        
        # Check for regular play indicators
        regular_play_indicators = [
            "weekly", "twice a week", "regularly", "every week",
            "few times a month", "often", "frequent"
        ]
        
        occasional_play_indicators = [
            "occasionally", "sometimes", "not often", "when i can",
            "here and there", "once in a while", "rarely"
        ]
        
        # Check for lesson indicators
        lesson_indicators = [
            "lessons", "coach", "instructor", "teaching", "coached",
            "take lessons", "have a coach", "work with"
        ]
        
        no_lesson_indicators = [
            "no lessons", "no coach", "never had lessons", "self taught",
            "just with friends", "on my own"
        ]
        
        has_regular_play = any(indicator in all_responses for indicator in regular_play_indicators)
        has_occasional_play = any(indicator in all_responses for indicator in occasional_play_indicators)
        has_lessons = any(indicator in all_responses for indicator in lesson_indicators)
        no_lessons = any(indicator in all_responses for indicator in no_lesson_indicators)
        
        # Decision logic for 1+ year players
        if has_regular_play and has_lessons:
            return "Intermediate"
        elif has_regular_play and not no_lessons:  # Regular play, lessons unclear
            return "Intermediate"
        elif has_lessons and not has_occasional_play:  # Has lessons, frequency unclear
            return "Intermediate"
        elif has_occasional_play and no_lessons:  # Occasional + no lessons
            return "Beginner"
        else:
            # When in doubt for 1+ year players, lean toward Intermediate
            # since they've stuck with it for over a year
            return "Intermediate"
    
    # STEP 5: Look for other experience indicators if no clear time mentioned
    experience_indicators = [
        "experience", "played before", "been playing", "familiar with",
        "know the basics", "comfortable with"
    ]
    
    if any(indicator in all_responses for indicator in experience_indicators):
        # Some experience mentioned but unclear - check for advanced concepts
        advanced_concepts = [
            "strategy", "tactics", "consistency", "power", "spin",
            "serve", "volley", "backhand", "forehand"
        ]
        
        if any(concept in all_responses for concept in advanced_concepts):
            return "Intermediate"
    
    # DEFAULT: When in doubt, return Beginner
    return "Beginner"


def handle_introduction_sequence(user_message: str, claude_client):
    """
    Enhanced introduction sequence with conversational level assessment
    """
    intro_state = st.session_state.get("intro_state", "waiting_for_name")
    
    if intro_state == "waiting_for_name":
        # Extract name from user response
        player_name = extract_name_from_response(user_message)
        if player_name:
            st.session_state.collected_name = player_name
            st.session_state.intro_state = "checking_if_new"
            return f"Nice to meet you, {player_name}! Are you pretty new to tennis?"
    
    elif intro_state == "checking_if_new":
        user_lower = user_message.lower().strip()
        
        # Check for clear "yes" responses to being new
        yes_responses = ["yes", "yeah", "yep", "sure", "i am", "pretty new", "very new", "just started"]
        if any(response in user_lower for response in yes_responses):
            # Clear beginner - update player and finish intro
            success = update_player_info(
                st.session_state.player_record_id,
                st.session_state.collected_name,
                "Beginner"
            )
            st.session_state.intro_completed = True
            st.session_state.intro_state = "complete"
            return "Perfect! Let's get started. What would you like to work on today?"
        
        # Check for clear "no" responses
        no_responses = ["no", "nope", "not really", "not new", "been playing"]
        if any(response in user_lower for response in no_responses):
            st.session_state.intro_state = "asking_time"
            return "How long have you been playing tennis?"
        
        # Ambiguous response - probe more
        st.session_state.intro_state = "asking_time"
        return "Tell me a bit about your tennis experience - how long have you been playing?"
    
    elif intro_state == "asking_time":
        user_lower = user_message.lower().strip()
        
        # Quick check for obvious beginner time indicators
        beginner_time_phrases = [
            "few months", "couple months", "just started", "not long",
            "recently", "6 months", "less than a year"
        ]
        
        if any(phrase in user_lower for phrase in beginner_time_phrases):
            # Clear beginner based on time
            success = update_player_info(
                st.session_state.player_record_id,
                st.session_state.collected_name,
                "Beginner"
            )
            st.session_state.intro_completed = True
            st.session_state.intro_state = "complete"
            return "Great! What would you like to work on today?"
        
        # Check for 1+ year indicators
        year_indicators = ["year", "years", "while", "long time"]
        if any(indicator in user_lower for indicator in year_indicators):
            st.session_state.intro_state = "asking_frequency"
            return "Nice! How often do you play? Do you take lessons?"
        
        # Unclear time response - ask for frequency anyway
        st.session_state.intro_state = "asking_frequency"  
        return "How often do you get to play? Do you take lessons or work with a coach?"
    
    elif intro_state == "asking_frequency":
        # Now we have enough for assessment
        assessed_level = assess_player_level_from_conversation(st.session_state.messages, claude_client)
        
        # Update player record with collected name and assessed level
        success = update_player_info(
            st.session_state.player_record_id,
            st.session_state.collected_name,
            assessed_level
        )
        
        st.session_state.intro_completed = True
        st.session_state.intro_state = "complete"
        
        # Acknowledge their level naturally
        if assessed_level == "Intermediate":
            return "Sounds like you've got some good experience! What's on your mind for today's session?"
        else:
            return "Perfect! What would you like to work on today?"
    
    return None

def setup_player_session_with_continuity(player_email: str):
    """
    Enhanced player setup with smart two-message welcome system
    """
    existing_player = find_player_by_email(player_email)
    
    if existing_player:
        # RETURNING PLAYER
        player_data = existing_player['fields']
        st.session_state.player_record_id = existing_player['id']
        st.session_state.is_returning_player = True
        player_name = player_data.get('name', 'there')
        session_number = player_data.get('total_sessions', 0) + 1
        
        # Load coaching history
        with st.spinner("Loading your coaching history..."):
            recent_summaries = get_player_recent_summaries(existing_player['id'], 2)
            st.session_state.coaching_history = recent_summaries
        
        # Generate two-part welcome message
        greeting, followup, should_delay = enhanced_generate_personalized_welcome_message(
            player_name, 
            session_number, 
            recent_summaries, 
            True
        )
        
        # Store followup message for delayed sending
        if should_delay and followup:
            st.session_state.pending_followup = followup
            st.session_state.followup_timer_start = time.time()
            st.session_state.followup_sent = False
        
        # Update session count
        update_player_session_count(existing_player['id'])
        
        # Store player info for coaching context
        st.session_state.player_name = player_name
        st.session_state.player_level = player_data.get('tennis_level', 'Beginner')
        
        return greeting
        
    else:
        # NEW PLAYER
        new_player = create_new_player(player_email, "", "")
        
        if new_player:
            st.session_state.player_record_id = new_player['id']
            st.session_state.is_returning_player = False
            st.session_state.coaching_history = []
            
            # Set introduction state for new players
            st.session_state.intro_state = "waiting_for_name"
            st.session_state.intro_completed = False
            
            # Clear any previous player info and pending messages
            st.session_state.player_name = ""
            st.session_state.player_level = ""
            st.session_state.pending_followup = None
            st.session_state.followup_sent = False
            
            return "Hi! I'm Coach TA, your personal tennis coach. What's your name?"
        else:
            st.error("Error creating player profile. Please try again.")
            return None

def check_and_send_followup_message():
    """
    Check if it's time to send the followup message (5-second timer logic)
    Returns: (should_send_followup, followup_message)
    """
    # Check if we have a pending followup
    if not st.session_state.get('pending_followup') or st.session_state.get('followup_sent'):
        return False, None
    
    # Check if 5 seconds have passed
    start_time = st.session_state.get('followup_timer_start', 0)
    current_time = time.time()
    
    if current_time - start_time >= 5.0:  # 5 seconds
        followup_msg = st.session_state.pending_followup
        
        # Mark as sent to prevent duplicate sending
        st.session_state.followup_sent = True
        st.session_state.pending_followup = None
        
        return True, followup_msg
    
    return False, None


def handle_user_response_during_timer():
    """
    Handle when user responds before the 5-second timer
    Returns: (should_send_followup_now, followup_message)
    """
    if st.session_state.get('pending_followup') and not st.session_state.get('followup_sent'):
        followup_msg = st.session_state.pending_followup
        
        # Mark as sent and clear pending
        st.session_state.followup_sent = True
        st.session_state.pending_followup = None
        
        return True, followup_msg
    
    return False, None

def log_welcome_messages(player_record_id: str, session_id: str, greeting_msg: str, followup_msg: str = None):
    """
    Log both welcome messages to the dual logging system
    """
    message_counter = 0
    
    # Log the greeting message
    message_counter += 1
    log_message_to_sss(player_record_id, session_id, message_counter, "assistant", greeting_msg)
    log_message_to_conversation_log(player_record_id, session_id, message_counter, "assistant", greeting_msg)
    
    # Log the followup message if provided
    if followup_msg:
        message_counter += 1
        log_message_to_sss(player_record_id, session_id, message_counter, "assistant", followup_msg)
        log_message_to_conversation_log(player_record_id, session_id, message_counter, "assistant", followup_msg)
    
    return message_counter

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

# Add these functions RIGHT BEFORE your main() function
# REPLACE your existing admin functions with these enhanced versions

# REPLACE all your existing admin functions with these updated versions

def get_all_coaching_sessions():
    """Fixed version - reads from Active_Sessions with actual resource data"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {
            "sort[0][field]": "timestamp",
            "sort[0][direction]": "desc",
            "maxRecords": 200
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return []
        
        records = response.json().get('records', [])
        
        # Group by session_id and calculate resource analytics from Active_Sessions
        sessions = {}
        for record in records:
            fields = record.get('fields', {})
            session_id = fields.get('session_id')
            
            if session_id:
                if session_id not in sessions:
                    sessions[session_id] = {
                        'session_id': session_id,
                        'message_count': 0,
                        'total_resources': 0,
                        'coach_responses': 0,
                        'timestamp': fields.get('timestamp', ''),
                        'status': fields.get('session_status', 'unknown')
                    }
                
                sessions[session_id]['message_count'] += 1
                
                # Get resource data from Active_Sessions (this table DOES have the data)
                if fields.get('role') == 'coach':
                    sessions[session_id]['coach_responses'] += 1
                    # Active_Sessions has coaching_resources_used field too!
                    resources_used = fields.get('coaching_resources_used', 0)
                    if resources_used:
                        sessions[session_id]['total_resources'] += resources_used
        
        # Calculate resource efficiency
        for session in sessions.values():
            if session['coach_responses'] > 0:
                session['resources_per_response'] = round(session['total_resources'] / session['coach_responses'], 1)
            else:
                session['resources_per_response'] = 0
        
        sessions_list = list(sessions.values())
        sessions_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return sessions_list
        
    except Exception as e:
        st.error(f"Error fetching sessions: {e}")
        return []

def get_conversation_messages_with_resources(session_id):
    """Fixed version - reads from Active_Sessions with proper chat bubbles and resource details"""
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
                    'resources_used': fields.get('coaching_resources_used', 0),
                    'resource_details': fields.get('resource_details', ''),
                    'log_id': fields.get('log_id', 0)
                })
            
            return messages
        return []
    except Exception as e:
        st.error(f"Error fetching conversation: {e}")
        return []

def display_resource_analytics(messages):
    """Display resource usage analytics for a session"""
    # Calculate analytics
    total_messages = len(messages)
    coach_messages = [m for m in messages if m['role'] == 'coach']
    player_messages = [m for m in messages if m['role'] == 'player']
    
    total_resources = sum(m.get('resources_used', 0) for m in coach_messages)
    responses_with_resources = len([m for m in coach_messages if m.get('resources_used', 0) > 0])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", total_messages)
    with col2:
        st.metric("Coach Responses", len(coach_messages))
    with col3:
        st.metric("Resources Used", total_resources)
    with col4:
        resource_rate = f"{(responses_with_resources/len(coach_messages)*100):.0f}%" if coach_messages else "0%"
        st.metric("Resource Usage Rate", resource_rate)
    
    # Resource breakdown
    if total_resources > 0:
        st.markdown("#### 📚 Resource Usage Breakdown")
        
        resource_responses = []
        for i, msg in enumerate(coach_messages):
            if msg.get('resources_used', 0) > 0:
                resource_responses.append({
                    'Response #': i + 1,
                    'Resources': msg['resources_used'],
                    'Response Preview': msg['content'][:80] + "..." if len(msg['content']) > 80 else msg['content']
                })
        
        if resource_responses:
            df = pd.DataFrame(resource_responses)
            st.dataframe(df, use_container_width=True)
            
            # Show detailed resource information
            st.markdown("#### 🔍 Detailed Resource Analysis")
            for i, msg in enumerate(coach_messages):
                if msg.get('resources_used', 0) > 0 and msg.get('resource_details'):
                    with st.expander(f"Response #{i+1}: {msg['resources_used']} resources used"):
                        st.markdown("**Coach Response:**")
                        st.write(msg['content'])
                        st.markdown("**Resources Used:**")
                        st.text(msg['resource_details'])

def get_all_players():
    """Fetch all players with their session counts and engagement metrics"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {
            "sort[0][field]": "total_sessions",
            "sort[0][direction]": "desc",
            "maxRecords": 100
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return []
        
        records = response.json().get('records', [])
        players = []
        
        for record in records:
            fields = record.get('fields', {})
            player_data = {
                'player_id': record['id'],
                'name': fields.get('name', 'Unknown'),
                'email': fields.get('email', ''),
                'tennis_level': fields.get('tennis_level', 'Not specified'),
                'total_sessions': fields.get('total_sessions', 0),
                'first_session_date': fields.get('first_session_date', ''),
                'player_status': fields.get('player_status', 'Unknown')
            }
            players.append(player_data)
        
        return players
        
    except Exception as e:
        st.error(f"Error fetching players: {e}")
        return []

def get_player_sessions_from_conversation_log(player_id: str):
    """Get all sessions for a specific player from Conversation_Log with detailed metrics - FIXED VERSION"""
    try:
        # First get player info
        player_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_id}"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        player_response = requests.get(player_url, headers=headers)
        if player_response.status_code != 200:
            return [], {}
        
        player_info = player_response.json().get('fields', {})
        
        # STEP 1: Get all Active_Sessions for this player to find their session_ids
        active_sessions_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        active_params = {
            "sort[0][field]": "timestamp",
            "sort[0][direction]": "desc", 
            "maxRecords": 500
        }
        
        active_response = requests.get(active_sessions_url, headers=headers, params=active_params)
        if active_response.status_code != 200:
            return [], player_info
            
        active_records = active_response.json().get('records', [])
        
        # Find session_ids for this player
        player_session_ids = set()
        session_id_to_record_id = {}  # Map session_id to Active_Sessions record_id
        
        for record in active_records:
            fields = record.get('fields', {})
            record_player_ids = fields.get('player_id', [])
            
            if isinstance(record_player_ids, list) and player_id in record_player_ids:
                session_id = fields.get('session_id')
                if session_id:
                    player_session_ids.add(session_id)
                    session_id_to_record_id[session_id] = record['id']
        
        if not player_session_ids:
            return [], player_info  # No sessions found for this player
        
        # STEP 2: Get all Conversation_Log records
        conv_log_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Conversation_Log"
        conv_params = {
            "sort[0][field]": "log_id",
            "sort[0][direction]": "desc",
            "maxRecords": 1000
        }
        
        conv_response = requests.get(conv_log_url, headers=headers, params=conv_params)
        if conv_response.status_code != 200:
            return [], player_info
        
        conv_records = conv_response.json().get('records', [])
        
        # STEP 3: Filter Conversation_Log records for this player's sessions
        session_metrics = {}
        
        for record in conv_records:
            fields = record.get('fields', {})
            record_session_links = fields.get('session_id', [])
            
            # Check if this conversation record links to any of our player's sessions
            for session_link in record_session_links:
                # session_link is the Active_Sessions record_id
                # Find the corresponding session_id number
                matching_session_id = None
                for sid, rid in session_id_to_record_id.items():
                    if rid == session_link:
                        matching_session_id = sid
                        break
                
                if matching_session_id and matching_session_id in player_session_ids:
                    if matching_session_id not in session_metrics:
                        session_metrics[matching_session_id] = {
                            'session_id': matching_session_id,
                            'message_count': 0,
                            'total_resources': 0,
                            'coach_responses': 0,
                            'player_responses': 0,
                            'first_log_id': float('inf'),
                            'last_log_id': 0,
                            'status': 'completed'
                        }
                    
                    session = session_metrics[matching_session_id]
                    session['message_count'] += 1
                    
                    # Track message types and resources
                    role = fields.get('role')
                    if role == 'coach':
                        session['coach_responses'] += 1
                        resources_used = fields.get('coaching_resources_used', 0)
                        if resources_used:
                            session['total_resources'] += resources_used
                    elif role == 'player':
                        session['player_responses'] += 1
                    
                    # Track log_id range for rough timing
                    log_id = fields.get('log_id', 0)
                    if log_id < session['first_log_id']:
                        session['first_log_id'] = log_id
                    if log_id > session['last_log_id']:
                        session['last_log_id'] = log_id
        
        # Calculate session efficiency metrics
        for session in session_metrics.values():
            if session['coach_responses'] > 0:
                session['resources_per_response'] = round(session['total_resources'] / session['coach_responses'], 1)
            else:
                session['resources_per_response'] = 0
            
            # Rough duration estimate based on log_id difference
            session['duration_minutes'] = max(1, (session['last_log_id'] - session['first_log_id']) * 0.1)
            session['first_message_time'] = str(session['first_log_id'])
        
        sessions_list = list(session_metrics.values())
        sessions_list.sort(key=lambda x: x['first_log_id'], reverse=True)
        
        return sessions_list, player_info
        
    except Exception as e:
        st.error(f"Error fetching player sessions: {e}")
        return [], {}

def display_player_engagement_analytics(sessions, player_info):
    """Display comprehensive player engagement analytics"""
    if not sessions:
        st.warning("No sessions found for this player.")
        return
    
    # Player overview metrics
    total_sessions = len(sessions)
    total_messages = sum(s['message_count'] for s in sessions)
    total_resources = sum(s['total_resources'] for s in sessions)
    total_duration = sum(s['duration_minutes'] for s in sessions)
    
    completed_sessions = len([s for s in sessions if s['status'] == 'completed'])
    avg_messages_per_session = total_messages / total_sessions if total_sessions > 0 else 0
    avg_duration = total_duration / total_sessions if total_sessions > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", total_sessions)
    with col2:
        st.metric("Completed Sessions", completed_sessions)
    with col3:
        st.metric("Avg Messages/Session", f"{avg_messages_per_session:.1f}")
    with col4:
        st.metric("Total Coaching Time", f"{avg_duration:.0f} min avg")
    
    # Resource usage metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Resources Used", total_resources)
    with col2:
        avg_resources = total_resources / sum(s['coach_responses'] for s in sessions) if sum(s['coach_responses'] for s in sessions) > 0 else 0
        st.metric("Avg Resources/Response", f"{avg_resources:.1f}")
    with col3:
        resource_sessions = len([s for s in sessions if s['total_resources'] > 0])
        resource_rate = (resource_sessions / total_sessions * 100) if total_sessions > 0 else 0
        st.metric("Sessions Using Resources", f"{resource_rate:.0f}%")
    
    # Session timeline
    st.markdown("#### 📅 Session History")
    session_data = []
    for i, session in enumerate(sessions):
        session_data.append({
            'Session #': len(sessions) - i,  # Most recent = highest number
            'Session ID': session['session_id'],
            'Messages': session['message_count'],
            'Resources': session['total_resources'],
            'Duration (min)': f"{session['duration_minutes']:.1f}",
            'Status': session['status'].title()
        })
    
    df = pd.DataFrame(session_data)
    st.dataframe(df, use_container_width=True)
    
    # Engagement trends
    if len(sessions) > 1:
        st.markdown("#### 📈 Engagement Trends")
        
        # Recent vs older sessions comparison
        recent_sessions = sessions[:3] if len(sessions) >= 3 else sessions
        older_sessions = sessions[3:6] if len(sessions) > 6 else sessions[len(recent_sessions):]
        
        if older_sessions:
            recent_avg_messages = sum(s['message_count'] for s in recent_sessions) / len(recent_sessions)
            older_avg_messages = sum(s['message_count'] for s in older_sessions) / len(older_sessions)
            
            message_trend = recent_avg_messages - older_avg_messages
            trend_emoji = "📈" if message_trend > 0 else "📉" if message_trend < 0 else "➡️"
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recent Sessions Avg", f"{recent_avg_messages:.1f} msgs", f"{message_trend:+.1f}")
            with col2:
                st.write(f"{trend_emoji} **Engagement Trend:** {'Increasing' if message_trend > 0 else 'Decreasing' if message_trend < 0 else 'Stable'}")

def display_admin_interface():
    """Enhanced admin interface reading from Active_Sessions for resource analytics"""
    st.title("🔧 Tennis Coach AI - Admin Interface")
    st.markdown("### Session Management & Player Analytics")
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 All Sessions", "👥 Player Engagement"])
    
    with tab1:
        # Session overview from Active_Sessions
        sessions = get_all_coaching_sessions()
        
        if not sessions:
            st.warning("No coaching sessions found.")
        else:
            st.markdown(f"**Found {len(sessions)} coaching sessions:**")
            
            # Summary analytics
            total_resources = sum(s['total_resources'] for s in sessions)
            total_responses = sum(s['coach_responses'] for s in sessions)
            avg_resources = total_resources / total_responses if total_responses > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", len(sessions))
            with col2:
                st.metric("Total Resources Used", total_resources)
            with col3:
                st.metric("Avg Resources/Response", f"{avg_resources:.1f}")
            
            st.markdown("---")
            
            # Session selector
            session_options = {}
            for session in sessions[:15]:
                timestamp = session['timestamp']
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%m/%d %H:%M")
                except:
                    formatted_time = "Unknown time"
                
                status_emoji = "✅" if session['status'] == 'completed' else "🟡"
                resource_info = f"📚{session['total_resources']}"
                display_name = f"{status_emoji} Session {session['session_id']} | {session['message_count']} msgs | {resource_info} | {formatted_time}"
                session_options[display_name] = session['session_id']
            
            selected_display = st.selectbox(
                "🎾 Select Session to Analyze",
                options=list(session_options.keys()),
                help="Choose a session to view conversation and resource analytics"
            )
            
            if selected_display:
                selected_session_id = session_options[selected_display]
                session_info = next(s for s in sessions if s['session_id'] == selected_session_id)
                
                # Display session metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Session ID", session_info['session_id'])
                with col2:
                    st.metric("Messages", session_info['message_count'])
                with col3:
                    st.metric("Resources Used", session_info['total_resources'])
                with col4:
                    st.metric("Resources/Response", session_info['resources_per_response'])
                
                st.markdown("---")
                
                # Get conversation with resource details - FIXED VERSION
                messages = get_conversation_messages_with_resources(selected_session_id)
                
                if messages:
                    # Create tabs for different views
                    conv_tab1, conv_tab2 = st.tabs(["💬 Conversation", "📊 Resource Analytics"])
                    
                    with conv_tab1:
                        st.markdown("### 💬 Conversation Log")
                        
                        for msg in messages:
                            role = msg['role']
                            content = msg['content']
                            resources_used = msg.get('resources_used', 0)
                            
                            if role == 'player':
                                # Player message - left side, blue bubble
                                st.markdown(f"""
                                <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                                    <div style="background-color: #E3F2FD; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #BBDEFB;">
                                        <strong>🧑‍🎓 Player:</strong><br>
                                        {content}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            elif role == 'coach':
                                # Coach message - right side, green bubble
                                resource_indicator = f" 📚 {resources_used}" if resources_used > 0 else ""
                                st.markdown(f"""
                                <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                                    <div style="background-color: #E8F5E8; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #C8E6C9;">
                                        <strong>🎾 Coach TA:</strong>{resource_indicator}<br>
                                        {content}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Show expandable resource details if available
                                if resources_used > 0 and msg.get('resource_details'):
                                    with st.expander(f"📊 View {resources_used} coaching resources"):
                                        st.text(msg['resource_details'])
                    
                    with conv_tab2:
                        # Resource analytics tab
                        display_resource_analytics(messages)
                        
                else:
                    st.warning("No messages found for this session.")
    
    with tab2:
        # Player engagement analysis
        st.markdown("### 👥 Player Engagement Analysis")
        
        players = get_all_players()
        
        if not players:
            st.warning("No players found in the database.")
        else:
            # Player selector
            player_options = {}
            for player in players:
                name = player['name'] if player['name'] != 'Unknown' else player['email'].split('@')[0]
                level = player['tennis_level']
                sessions_count = player['total_sessions']
                display_name = f"{name} ({level}) - {sessions_count} sessions"
                player_options[display_name] = player['player_id']
            
            selected_player_display = st.selectbox(
                "🧑‍🎓 Select Player to Analyze",
                options=list(player_options.keys()),
                help="Choose a player to view their complete engagement history"
            )
            
            if selected_player_display:
                selected_player_id = player_options[selected_player_display]
                
                # Get player sessions and info
                player_sessions, player_info = get_player_sessions_from_conversation_log(selected_player_id)
                
                if player_sessions:
                    # Player info header
                    st.markdown("#### 🧑‍🎓 Player Profile")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Name:** {player_info.get('name', 'Unknown')}")
                        st.write(f"**Email:** {player_info.get('email', 'Unknown')}")
                    with col2:
                        st.write(f"**Tennis Level:** {player_info.get('tennis_level', 'Not specified')}")
                        st.write(f"**Status:** {player_info.get('player_status', 'Unknown')}")
                    with col3:
                        try:
                            first_session = datetime.fromisoformat(player_info.get('first_session_date', '').replace('Z', '+00:00')).strftime("%m/%d/%Y")
                        except:
                            first_session = "Unknown"
                        st.write(f"**First Session:** {first_session}")
                        st.write(f"**Total Sessions:** {player_info.get('total_sessions', 0)}")
                    
                    st.markdown("---")
                    
                    # Player engagement analytics
                    display_player_engagement_analytics(player_sessions, player_info)
                    
                    st.markdown("---")
                    
                    # Individual session selector for this player
                    st.markdown("#### 🔍 View Individual Sessions")
                    session_options = {}
                    for i, session in enumerate(player_sessions):
                        status_emoji = "✅" if session['status'] == 'completed' else "🟡"
                        resource_info = f"📚{session['total_resources']}"
                        display_name = f"{status_emoji} Session #{len(player_sessions)-i} | {session['session_id']} | {session['message_count']} msgs | {resource_info}"
                        session_options[display_name] = session['session_id']
                    
                    if session_options:
                        selected_session_display = st.selectbox(
                            "Select a session to view details:",
                            options=list(session_options.keys()),
                            key="player_session_selector"
                        )
                        
                        if selected_session_display:
                            selected_session_id = session_options[selected_session_display]
                            messages = get_conversation_messages_with_resources(selected_session_id)
                            
                            if messages:
                                st.markdown("##### 💬 Session Conversation")
                                for msg in messages:
                                    role = msg['role']
                                    content = msg['content']
                                    resources_used = msg.get('resources_used', 0)
                                    
                                    if role == 'player':
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: flex-start; margin: 10px 0;">
                                            <div style="background-color: #E3F2FD; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #BBDEFB;">
                                                <strong>🧑‍🎓 Player:</strong><br>
                                                {content}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif role == 'coach':
                                        resource_indicator = f" 📚 {resources_used}" if resources_used > 0 else ""
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                                            <div style="background-color: #E8F5E8; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #C8E6C9;">
                                                <strong>🎾 Coach TA:</strong>{resource_indicator}<br>
                                                {content}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        if resources_used > 0 and msg.get('resource_details'):
                                            with st.expander(f"📊 View {resources_used} coaching resources"):
                                                st.text(msg['resource_details'])
                else:
                    st.warning("No sessions found for this player.")
    
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
    
    # CHECK FOR ADMIN MODE FIRST
    if st.session_state.get('admin_mode', False):
        display_admin_interface()
        return
    
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
            st.session_state.pending_followup = None
            st.session_state.followup_sent = False
            st.session_state.recent_greetings = []
            st.rerun()
    
    # PLAYER SETUP FORM
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
                        
                        # Add the greeting message
                        st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                        
                        # Log the initial welcome message
                        if st.session_state.get("player_record_id"):
                            st.session_state.message_counter = 1
                            log_message_to_sss(
                                st.session_state.player_record_id,
                                session_id,
                                st.session_state.message_counter,
                                "assistant",
                                welcome_msg
                            )
                            log_message_to_conversation_log(
                                st.session_state.player_record_id,
                                session_id,
                                st.session_state.message_counter,
                                "assistant",
                                welcome_msg
                            )
                        
                        st.success("Welcome! Ready to start your coaching session.")
                        st.rerun()
        return
    
    # CHECK FOR PENDING FOLLOWUP MESSAGE (5-second timer)
    should_send_followup, followup_message = check_and_send_followup_message()
    if should_send_followup and followup_message:
        # Add followup message to conversation
        st.session_state.messages.append({"role": "assistant", "content": followup_message})
        
        # Log the followup message
        if st.session_state.get("player_record_id"):
            st.session_state.message_counter += 1
            log_message_to_sss(
                st.session_state.player_record_id,
                st.session_state.session_id,
                st.session_state.message_counter,
                "assistant",
                followup_message
            )
            log_message_to_conversation_log(
                st.session_state.player_record_id,
                st.session_state.session_id,
                st.session_state.message_counter,
                "assistant",
                followup_message
            )
        
        st.rerun()  # Refresh to show the followup message
    
    # DISPLAY CONVERSATION MESSAGES
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # USER INPUT HANDLING
    if prompt := st.chat_input("Ask your tennis coach..."):
        # ADMIN MODE TRIGGER
        if prompt.strip().lower() == "hilly spike":
            st.session_state.admin_mode = True
            st.rerun()
            return
        
        # CHECK IF USER RESPONDED DURING TIMER - SEND FOLLOWUP NOW
        should_send_immediate_followup, immediate_followup = handle_user_response_during_timer()
        if should_send_immediate_followup and immediate_followup:
            # Insert followup message before processing user input
            with st.chat_message("assistant"):
                st.markdown(immediate_followup)
            
            st.session_state.messages.append({"role": "assistant", "content": immediate_followup})
            
            # Log the followup message
            if st.session_state.get("player_record_id"):
                st.session_state.message_counter += 1
                log_message_to_sss(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    immediate_followup
                )
                log_message_to_conversation_log(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    immediate_followup
                )
        
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
        
        # DUAL LOGGING: Log user message to both tables
        if st.session_state.get("player_record_id"):
            log_message_to_sss(
                st.session_state.player_record_id,
                st.session_state.session_id,
                st.session_state.message_counter,
                "user",
                prompt
            )
            log_message_to_conversation_log(
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
                
                # DUAL LOGGING: Log intro response to both tables
                if st.session_state.get("player_record_id"):
                    log_message_to_sss(
                        st.session_state.player_record_id,
                        st.session_state.session_id,
                        st.session_state.message_counter,
                        "assistant",
                        intro_response
                    )
                    log_message_to_conversation_log(
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
            
            # DUAL LOGGING: Log confirmation message to both tables
            if st.session_state.get("player_record_id"):
                log_message_to_sss(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    confirmation_msg
                )
                log_message_to_conversation_log(
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
                
                # DUAL LOGGING: Log closing response to both tables
                if st.session_state.get("player_record_id"):
                    log_message_to_sss(
                        st.session_state.player_record_id,
                        st.session_state.session_id,
                        st.session_state.message_counter,
                        "assistant",
                        closing_response
                    )
                    log_message_to_conversation_log(
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
                    
                    # DUAL LOGGING: Log coach response with chunks to both tables
                    if st.session_state.get("player_record_id"):
                        log_message_to_sss(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
                            st.session_state.message_counter,
                            "assistant",
                            response,
                            chunks
                        )
                        log_message_to_conversation_log(
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
                    
                    # DUAL LOGGING: Log error message to both tables
                    if st.session_state.get("player_record_id"):
                        log_message_to_sss(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
                            st.session_state.message_counter,
                            "assistant",
                            error_msg
                        )
                        log_message_to_conversation_log(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
                            st.session_state.message_counter,
                            "assistant",
                            error_msg
                        )

if __name__ == "__main__":
    main()
