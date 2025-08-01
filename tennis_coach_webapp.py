import streamlit as st
import os
import json
from typing import List, Dict
import time
import pandas as pd          # NEW
from datetime import datetime # NEW
import re

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

# Add the RAG sandbox import here
# Test import



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

def get_coaching_personality_enhancement():
    return """
COACHING BEHAVIOR ANCHORS:
- Acknowledge feelings first: "That sounds frustrating..." "I hear you saying..."
- Use coaching stories occasionally: "I had a player who..." "I remember working with someone who..." "In my experience..." (max 1 sentence)
- Share learning wisdom: Acknowledge that practice, unlearning, focus, and repetition are challenging but necessary
- Use smooth transitions: "That makes sense..." "Here's what helped..." "Let's try this..."
- Show protégé effect moments: "Your questions are making me think about this differently" "Teaching this helps me too"
- For brief responses: acknowledge + assume + ask follow-up
- Ask one specific follow-up question
- Keep responses conversational and supportive
- Balance direct advice with occasional brief stories

CRITICAL: NEVER include any meta-commentary, coach's notes, internal thoughts, or explanations about your coaching approach in your response. Only provide the direct coaching advice to the player.
"""

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
        for msg in conversation_history[12:]:
            role = "Player" if msg['role'] == 'user' else "Coach"
            history_text += f"{role}: {msg['content']}\n"
    return f"""You are a professional tennis coach providing REMOTE coaching advice through chat. The player is not physically with you, so focus on guidance they can apply on their own.

Guidelines:
- CRITICAL: If your response would naturally be 3+ sentences, split into exactly 2 consecutive messages
- Send both messages immediately, one after another
- Message 1: Core concept - 1-2 sentences maximum
- Message 2: Application + follow-up question - 1-2 sentences maximum
- If you have more information, ask if they want to know more about specific aspects
- Focus on ONE specific tip or correction per message pair
- Give advice for SOLO practice or general technique improvement
- Ask one engaging follow-up question to continue the conversation
- Use encouraging, supportive tone
- Be direct and actionable

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
        
        # Use provided name, or extract from email, or leave empty for Coach Taai collection
        if name:
            player_name = name
        else:
            # For new players, leave empty - Coach Taai will collect it
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

def classify_ending_intent(message_content: str) -> str:
    """
    Use AI to classify if a message indicates session ending intent
    Returns: 'DEFINITIVE', 'LIKELY', 'AMBIGUOUS', or 'NOT_ENDING'
    """
    try:
        # Quick obvious check first (for speed)
        obvious_definitive = ["end session", "stop session", "goodbye", "farewell"]
        if any(phrase in message_content.lower() for phrase in obvious_definitive):
            return "DEFINITIVE"
        
        # Use Claude for nuanced detection
        classification_prompt = f"""
Classify this message from a tennis coaching session. The player might be trying to end the session.

Player message: "{message_content}"

Classify as exactly one of these:

DEFINITIVE - Clear, unambiguous ending (like "goodbye coach", "end session", "bye coach")

LIKELY - Polite endings with gratitude or departure signals (like "thanks coach", "okay bye thank you", "see you soon coach", "got it, thanks coach")

AMBIGUOUS - Single words or unclear intent (like just "thanks", "bye", "okay", "done" by themselves)

NOT_ENDING - Clearly continuing conversation (questions, requests for help, tennis discussion)

Respond with only one word: DEFINITIVE, LIKELY, AMBIGUOUS, or NOT_ENDING"""

        # Get Claude's classification
        _, claude_client = setup_connections()
        if claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            
            classification = response.content[0].text.strip().upper()
            
            # Validate response
            valid_responses = ["DEFINITIVE", "LIKELY", "AMBIGUOUS", "NOT_ENDING"]
            if classification in valid_responses:
                return classification
        
        # Fallback classification if Claude fails
        return fallback_classification(message_content)
        
    except Exception as e:
        # Use fallback if anything goes wrong
        return fallback_classification(message_content)

def fallback_classification(message_content: str) -> str:
    """Simple fallback classification if AI fails"""
    message_lower = message_content.lower().strip()
    
    # Simple keyword-based fallback
    if any(word in message_lower for word in ["goodbye", "farewell", "end session"]):
        return "DEFINITIVE"
    elif any(word in message_lower for word in ["thanks coach", "bye coach", "see you"]):
        return "LIKELY" 
    elif any(word in message_lower for word in ["thanks", "bye", "okay", "done"]):
        return "AMBIGUOUS"
    else:
        return "NOT_ENDING"

def detect_session_end(message_content: str, conversation_history: list = None) -> dict:
    """
    Intelligent session end detection with context awareness
    Returns: {'should_end': bool, 'confidence': str, 'needs_confirmation': bool}
    """
    message_lower = message_content.lower().strip()
    
    # Use AI to classify the message intent
    ending_classification = classify_ending_intent(message_content)
    
    if ending_classification == "DEFINITIVE":
        return {'should_end': True, 'confidence': 'high', 'needs_confirmation': False}
    
    elif ending_classification == "LIKELY":
        return {'should_end': True, 'confidence': 'medium', 'needs_confirmation': True}
    
    elif ending_classification == "AMBIGUOUS":
        # Use existing context logic for ambiguous cases
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

def cleanup_abandoned_sessions(claude_client, dry_run=True, preview_mode=False):
    """Mark old active sessions as completed and generate summaries"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        # Find sessions older than 30 minutes that are still "active"
        from datetime import datetime, timedelta
        cutoff_time = (datetime.now() - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        params = {
            "filterByFormula": f"AND({{session_status}} = 'active', {{timestamp}} < '{cutoff_time}')",
            "sort[0][field]": "session_id",
            "sort[0][direction]": "desc"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            st.error(f"Failed to fetch sessions: {response.status_code}")
            return False
        
        all_abandoned_records = response.json().get('records', [])
        
        # Group messages by session_id and filter out admin sessions
        session_groups = {}
        admin_sessions_skipped = 0
        
        for record in all_abandoned_records:
            fields = record.get('fields', {})
            session_id = fields.get('session_id')
            message_content = fields.get('message_content', '')
            role = fields.get('role', '')
            message_order = fields.get('message_order', 0)
            
            if not session_id:
                continue
                
            # Skip obvious admin sessions
            if 'hilly spike' in message_content.lower():
                admin_sessions_skipped += 1
                continue
            
            # Group by session_id
            if session_id not in session_groups:
                session_groups[session_id] = {
                    'session_id': session_id,
                    'player_ids': fields.get('player_id', []),
                    'message_count': 0,
                    'first_timestamp': fields.get('timestamp', ''),
                    'messages': []
                }
            
            session_groups[session_id]['message_count'] += 1
            session_groups[session_id]['messages'].append({
                'role': role,
                'content': message_content,
                'order': message_order
            })
        
        # Sort messages within each session
        for session_data in session_groups.values():
            session_data['messages'].sort(key=lambda x: x['order'])
        
        # Filter out sessions that are likely admin (less than 4 messages)
        legitimate_sessions = []
        for session_id, session_data in session_groups.items():
            if session_data['message_count'] >= 4:  # Raised threshold
                legitimate_sessions.append(session_data)
            else:
                admin_sessions_skipped += 1
        
        if preview_mode:
            st.write(f"**Preview Mode: {len(legitimate_sessions)} sessions to review**")
            st.write(f"Skipped {admin_sessions_skipped} admin/short sessions")
            
            # Show detailed preview of each session
            for i, session_data in enumerate(legitimate_sessions):
                session_id = session_data['session_id']
                message_count = session_data['message_count']
                timestamp = session_data['first_timestamp']
                
                with st.expander(f"📋 Session {session_id} - {message_count} messages - {timestamp}"):
                    st.write("**Conversation Preview:**")
                    
                    # Show first few messages to determine if it's legitimate
                    for j, msg in enumerate(session_data['messages'][:6]):  # Show first 6 messages
                        role_label = "🧑‍🎓 Player" if msg['role'] == 'player' else "🎾 Coach"
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        st.write(f"**{role_label}:** {content}")
                        
                        if j == 5 and len(session_data['messages']) > 6:
                            st.write(f"... and {len(session_data['messages']) - 6} more messages")
                    
                    # Add cleanup button for this specific session
                    if st.button(f"🧹 Clean Up Session {session_id}", key=f"cleanup_{session_id}"):
                        with st.spinner(f"Cleaning up session {session_id}..."):
                            if session_data['player_ids']:
                                player_id = session_data['player_ids'][0]
                                session_marked = mark_session_completed(player_id, str(session_id))
                                
                                if session_marked:
                                    summary_created = process_completed_session(player_id, str(session_id), claude_client)
                                    if summary_created:
                                        st.success(f"✅ Successfully cleaned up session {session_id}")
                                    else:
                                        st.warning(f"⚠️ Session {session_id} marked complete but summary failed")
                                else:
                                    st.error(f"❌ Failed to mark session {session_id} as complete")
            
            return True
        
        if dry_run:
            st.write(f"DRY RUN: Would clean up {len(legitimate_sessions)} legitimate abandoned sessions")
            st.write(f"Skipped {admin_sessions_skipped} admin/short sessions")
            
            # Show summary of what would be cleaned up
            for session_data in legitimate_sessions:
                session_id = session_data['session_id']
                message_count = session_data['message_count']
                timestamp = session_data['first_timestamp']
                st.write(f"- Session {session_id}: {message_count} messages from {timestamp}")
                
            return True
            
        # Actual cleanup code only runs when dry_run=False
        cleanup_count = 0
        for session_data in legitimate_sessions:
            session_id = session_data['session_id']
            player_ids = session_data['player_ids']
            
            if session_id and player_ids:
                player_id = player_ids[0]
                
                # Mark session as completed
                session_marked = mark_session_completed(player_id, str(session_id))
                
                if session_marked:
                    # Generate summary for the abandoned session
                    summary_created = process_completed_session(player_id, str(session_id), claude_client)
                    if summary_created:
                        cleanup_count += 1
                        st.write(f"✅ Cleaned up session {session_id} ({session_data['message_count']} messages)")
                    else:
                        st.write(f"⚠️ Session {session_id} marked complete but summary failed")
                else:
                    st.write(f"❌ Failed to mark session {session_id} as complete")
        
        st.write(f"Successfully cleaned up {cleanup_count} legitimate abandoned sessions")
        st.write(f"Skipped {admin_sessions_skipped} admin/short sessions")
        return True
        
    except Exception as e:
        st.error(f"Cleanup error: {e}")
        return False

def analyze_session_fallback_details(session_id):
    """Get detailed fallback analysis for a specific session"""
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
            
            fallback_analysis = []
            for record in records:
                fields = record.get('fields', {})
                if fields.get('role') == 'coach':
                    message_content = fields.get('message_content', '')
                    resources_used = fields.get('coaching_resources_used', 0)
                    resource_details = fields.get('resource_details', '')
                    message_order = fields.get('message_order', 0)
                    
                    # Determine mode used
                    if resources_used > 0:
                        # Extract relevance from resource details
                        relevance_scores = []
                        if resource_details:
                            import re
                            scores = re.findall(r'(\d+\.\d+)\s+relevance', resource_details)
                            relevance_scores = [float(score) for score in scores]
                        
                        max_relevance = max(relevance_scores) if relevance_scores else 0.0
                        mode_used = "✅ Pinecone"
                        mode_details = f"(relevance: {max_relevance:.2f})"
                    else:
                        mode_used = "⚠️ Fallback"
                        mode_details = "(Claude-only)"
                    
                    fallback_analysis.append({
                        'message_order': message_order,
                        'message_preview': message_content[:60] + "..." if len(message_content) > 60 else message_content,
                        'mode_used': mode_used,
                        'mode_details': mode_details,
                        'resources_used': resources_used,
                        'resource_details': resource_details,
                        'relevance_scores': relevance_scores if 'relevance_scores' in locals() else []
                    })
            
            return fallback_analysis
        return []
        
    except Exception as e:
        st.error(f"Error analyzing session: {e}")
        return []

def detect_content_gaps():
    """Analyze fallback patterns to identify content gaps"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        # Get recent sessions (last 100 coach responses)
        params = {
            "filterByFormula": "{{role}} = 'coach'",
            "sort[0][field]": "timestamp",
            "sort[0][direction]": "desc",
            "maxRecords": 100
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            
            # Analyze fallback patterns
            fallback_topics = []
            high_relevance_topics = []
            total_responses = 0
            fallback_count = 0
            
            for record in records:
                fields = record.get('fields', {})
                resources_used = fields.get('coaching_resources_used', 0)
                resource_details = fields.get('resource_details', '')
                
                # Get corresponding user message to analyze topic
                session_id = fields.get('session_id')
                message_order = fields.get('message_order', 0)
                
                # Find the user message that triggered this response
                user_message = get_user_message_for_response(session_id, message_order - 1)
                
                total_responses += 1
                
                if resources_used == 0:
                    # This was a fallback
                    fallback_count += 1
                    if user_message:
                        topic_keywords = extract_topic_keywords(user_message)
                        fallback_topics.append({
                            'user_query': user_message[:50] + "..." if len(user_message) > 50 else user_message,
                            'keywords': topic_keywords,
                            'session_id': session_id
                        })
                else:
                    # This used Pinecone successfully
                    if user_message:
                        import re
                        scores = re.findall(r'(\d+\.\d+)\s+relevance', resource_details)
                        if scores:
                            max_relevance = max(float(score) for score in scores)
                            if max_relevance >= 0.8:  # High relevance
                                topic_keywords = extract_topic_keywords(user_message)
                                high_relevance_topics.append({
                                    'user_query': user_message[:50] + "..." if len(user_message) > 50 else user_message,
                                    'keywords': topic_keywords,
                                    'relevance': max_relevance,
                                    'session_id': session_id
                                })
            
            # Calculate fallback rate
            fallback_rate = (fallback_count / total_responses * 100) if total_responses > 0 else 0
            
            # Analyze topic patterns
            fallback_keywords = {}
            for topic in fallback_topics:
                for keyword in topic['keywords']:
                    fallback_keywords[keyword] = fallback_keywords.get(keyword, 0) + 1
            
            high_relevance_keywords = {}
            for topic in high_relevance_topics:
                for keyword in topic['keywords']:
                    high_relevance_keywords[keyword] = high_relevance_keywords.get(keyword, 0) + 1
            
            return {
                'fallback_rate': fallback_rate,
                'total_responses': total_responses,
                'fallback_count': fallback_count,
                'common_fallback_topics': sorted(fallback_keywords.items(), key=lambda x: x[1], reverse=True)[:10],
                'high_performing_topics': sorted(high_relevance_keywords.items(), key=lambda x: x[1], reverse=True)[:10],
                'recent_fallbacks': fallback_topics[:5],
                'recent_successes': high_relevance_topics[:5]
            }
            
        return None
        
    except Exception as e:
        st.error(f"Error detecting content gaps: {e}")
        return None

def get_user_message_for_response(session_id, expected_order):
    """Get the user message that triggered a specific coach response"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        params = {
            "filterByFormula": f"AND({{session_id}} = {session_id}, {{message_order}} = {expected_order}, {{role}} = 'player')",
            "maxRecords": 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                return records[0].get('fields', {}).get('message_content', '')
        return None
        
    except Exception as e:
        return None

def extract_topic_keywords(message):
    """Extract tennis-related keywords from a message"""
    tennis_keywords = [
        'forehand', 'backhand', 'serve', 'volley', 'smash', 'drop shot',
        'slice', 'topspin', 'backspin', 'grip', 'stance', 'footwork',
        'court', 'net', 'baseline', 'rally', 'match', 'game', 'set',
        'technique', 'practice', 'drill', 'training', 'coach', 'lesson',
        'mental', 'strategy', 'tactics', 'consistency', 'power', 'spin',
        'movement', 'positioning', 'timing', 'rhythm', 'balance'
    ]
    
    message_lower = message.lower()
    found_keywords = []
    
    for keyword in tennis_keywords:
        if keyword in message_lower:
            found_keywords.append(keyword)
    
    # If no tennis keywords found, extract first few words as general topic
    if not found_keywords:
        words = message_lower.split()[:3]
        found_keywords = [word for word in words if len(word) > 2]
    
    return found_keywords[:5]  # Return max 5 keywords

def mark_session_reviewed(session_id: str, admin_identifier: str = "admin") -> bool:
    """Mark a session as reviewed by admin"""
    try:
        # We'll store review status in the session state and also try to persist it
        if 'reviewed_sessions' not in st.session_state:
            st.session_state.reviewed_sessions = set()
        
        st.session_state.reviewed_sessions.add(session_id)
        
        # Try to also store in a persistent way using Airtable
        # We'll add a comment or note to one of the session records
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        # Find a record from this session to add review marker
        params = {
            "filterByFormula": f"{{session_id}} = {session_id}",
            "maxRecords": 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                record_id = records[0]['id']
                
                # Add review marker to the record
                update_url = f"{url}/{record_id}"
                update_headers = {
                    "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
                    "Content-Type": "application/json"
                }
                
                # Add or update a review field - we'll use resource_details field to store review info
                current_details = records[0].get('fields', {}).get('resource_details', '')
                review_marker = f"\n[ADMIN_REVIEWED: {admin_identifier} on {datetime.now().strftime('%Y-%m-%d %H:%M')}]"
                
                update_data = {
                    "fields": {
                        "resource_details": current_details + review_marker
                    }
                }
                
                requests.patch(update_url, headers=update_headers, json=update_data)
        
        return True
        
    except Exception as e:
        return False

def is_session_reviewed(session_id: str) -> bool:
    """Check if a session has been reviewed by admin"""
    try:
        # Check session state first
        if 'reviewed_sessions' not in st.session_state:
            st.session_state.reviewed_sessions = set()
        
        if session_id in st.session_state.reviewed_sessions:
            return True
        
        # Check database for persistent review marker
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        params = {
            "filterByFormula": f"{{session_id}} = {session_id}",
            "maxRecords": 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                resource_details = records[0].get('fields', {}).get('resource_details', '')
                if '[ADMIN_REVIEWED:' in resource_details:
                    # Add to session state for faster future checks
                    st.session_state.reviewed_sessions.add(session_id)
                    return True
        
        return False
        
    except Exception as e:
        return False

def get_review_status(session_id: str) -> dict:
    """Get detailed review status for a session"""
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        params = {
            "filterByFormula": f"{{session_id}} = {session_id}",
            "maxRecords": 1
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            records = response.json().get('records', [])
            if records:
                resource_details = records[0].get('fields', {}).get('resource_details', '')
                
                if '[ADMIN_REVIEWED:' in resource_details:
                    # Extract review info
                    import re
                    review_match = re.search(r'\[ADMIN_REVIEWED: (.*?) on (.*?)\]', resource_details)
                    if review_match:
                        return {
                            'reviewed': True,
                            'reviewer': review_match.group(1),
                            'review_date': review_match.group(2)
                        }
        
        return {'reviewed': False, 'reviewer': None, 'review_date': None}
        
    except Exception as e:
        return {'reviewed': False, 'reviewer': None, 'review_date': None}

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

# ENHANCED: Welcome message generation with better context
def enhanced_generate_personalized_welcome_message(player_name: str, session_number: int, recent_summaries: list, is_returning: bool) -> tuple:
    """
    Generate two-part welcome message: greeting + follow-up
    Returns: (greeting_message, followup_message) - both sent immediately
    """
    if not is_returning or not recent_summaries:
        # NEW PLAYER - single message
        return ("Hi! I'm Coach Taai, your personal tennis coach. What's your name?", None)
    
    # RETURNING PLAYER - two message system
    last_session = recent_summaries[0] if recent_summaries else {}
    
    # Calculate context
    player_record_id = st.session_state.get('player_record_id', '')
    days_since = calculate_days_since_last_session(player_record_id)
    session_tone = analyze_session_tone(last_session)
    
    # Generate both messages
    greeting = generate_smart_greeting(player_name, days_since, session_tone, session_number)
    followup = generate_followup_message(player_name, last_session, session_tone)
    
    return (greeting, followup)

# ENHANCED: Build conversational prompt with coaching history
def build_conversational_prompt_with_history(user_question: str, context_chunks: list, conversation_history: list, coaching_history: list = None, player_name: str = None, player_level: str = None) -> str:
    """Build Claude prompt with proper player context and memory"""
    
    # Check if this is intro
    is_intro = not st.session_state.get("intro_completed", True)
    
    if is_intro:
        # NEW PLAYER INTRODUCTION PROMPT
        intro_prompt = f"""You are Coach Taai. Be natural and conversational.

{get_coaching_personality_enhancement()}

INTRODUCTION FLOW:
- Start: "Hi! I'm Coach Taai, your personal tennis coach. What's your name?"
- After name: "Nice to meet you, [Name]! I am excited, tell me about your tennis. You been playing long?"
- After experience: "What's challenging you most on court right now?"
- Then transition: "Great! How about we work on [specific area] today?"

Keep responses SHORT (1-2 sentences max). Be enthusiastic but concise."""
        
        # Add current conversation context for intro
        history_text = ""
        if conversation_history:
            history_text = "\nCurrent conversation:\n"
            for msg in conversation_history[-20:]:  # Last 20 exchanges
                role = "Player" if msg['role'] == 'user' else "Coach Taai"
                history_text += f"{role}: {msg['content']}\n"
        
        # Clean context chunks of debug text
        cleaned_chunks = []
        for chunk in context_chunks:
            if chunk.get('text'):
                content_text = chunk.get('text', '')
                
                # Remove debug patterns
                debug_patterns = [
                    "Wait for player response before giving specific drill instruction",
                    "PATTERN 1", "PATTERN 2", "PATTERN 3",
                    "Internal note:", "Coach instruction:",
                    "DEBUG:", "Note to coach:", "Meta-commentary:",
                    "[Debug]", "[Internal]", "Coach note:",
                    "Wait for", "Before giving specific"
                ]
                
                for pattern in debug_patterns:
                    content_text = content_text.replace(pattern, "").strip()
                
                # Only include if there's meaningful content left
                if len(content_text.strip()) > 10:
                    cleaned_chunks.append(content_text)

        context_text = "\n\n".join(cleaned_chunks)
        
        return f"""{intro_prompt}
{history_text}

Tennis Knowledge: {context_text}

Player says: "{user_question}"

Respond naturally as Coach Taai:"""
    
    else:
        # REGULAR COACHING PROMPT WITH FULL CONTEXT
        player_context = ""
        if player_name and player_level:
            player_context = f"Player: {player_name} (Level: {player_level})\n"
        
        coaching_prompt = f"""You are Coach Taai coaching {player_name or 'the player'}.

{get_coaching_personality_enhancement()}

{player_context}

You provide direct, actionable tennis coaching advice. 

COACHING APPROACH:
- Ask 1 specific question about their situation
- Give ONE specific tip or drill appropriate for {player_level or 'their current'} level  
- If your complete advice would naturally be 3+ sentences, split into exactly 2 consecutive messages
- Send both messages immediately, one after another
- Message 1: Core concept (technique/explanation) - 1-2 sentences
- Message 2: Application (drill/practice method) + follow-up question - 1-2 sentences
- If you have even more information, end Message 2 with: "Want me to explain more about [specific aspect]?"
- Continue with more detail if player shows any interest (yes/sure/tell me more/questions about the topic)
- If player changes topics or asks different questions, follow their lead instead
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
            for msg in conversation_history[-20:]:  # Last 20 exchanges to maintain context
                role = "Player" if msg['role'] == 'user' else "Coach Taai"
                history_text += f"{role}: {msg['content']}\n"
        
        # Clean context chunks of debug text
        cleaned_chunks = []
        for chunk in context_chunks:
            if chunk.get('text'):
                content_text = chunk.get('text', '')
                
                # Remove debug patterns
                debug_patterns = [
                    "Wait for player response before giving specific drill instruction",
                    "PATTERN 1", "PATTERN 2", "PATTERN 3",
                    "Internal note:", "Coach instruction:",
                    "DEBUG:", "Note to coach:", "Meta-commentary:",
                    "[Debug]", "[Internal]", "Coach note:",
                    "Wait for", "Before giving specific"
                ]
                
                for pattern in debug_patterns:
                    content_text = content_text.replace(pattern, "").strip()
                
                # Only include if there's meaningful content left
                if len(content_text.strip()) > 10:
                    cleaned_chunks.append(content_text)

        context_text = "\n\n".join(cleaned_chunks)
        
        return f"""{coaching_prompt}
{history_text}

Tennis Knowledge: {context_text}

Player says: "{user_question}"

Give direct coaching advice:"""

def get_smart_coaching_response(prompt, index, claude_client, coaching_mode, top_k):
    """
    Smart coaching response with three modes:
    - Auto: Pinecone+Claude with fallback to Claude-only if relevance < admin-set threshold (default 0.45)
    - Pinecone+Claude: Always use Pinecone
    - Claude Only: Never use Pinecone
    """
    
    # Get player context
    coaching_history = st.session_state.get('coaching_history', [])
    player_name, player_level = get_current_player_info(st.session_state.get("player_record_id", ""))
    
    # Claude Only Mode
    if coaching_mode == "🧠 Claude Only":
        st.session_state.last_coaching_mode_used = "🧠 Claude-only mode active"
        
        # Build Claude-only prompt
        recent_conversation = ""
        if len(st.session_state.messages) > 1:
            recent_conversation = "\nCURRENT SESSION CONVERSATION:\n"
            for msg in st.session_state.messages[20:]:
                role = "Player" if msg['role'] == 'user' else "Coach Taai"
                recent_conversation += f"{role}: {msg['content']}\n"
        
        session_context = ""
        if coaching_history and len(coaching_history) > 0 and len(st.session_state.messages) <= 4:
            last_session = coaching_history[0]
            if last_session.get('technical_focus'):
                session_context = f"\nPrevious session focus: {last_session['technical_focus']}"
                session_context += f"\nNOTE: Focus on current conversation topic, not previous session topics."
        
        claude_only_prompt = f"""You are Coach Taai, a professional tennis coach providing remote coaching advice through chat.

{get_coaching_personality_enhancement()}

Player: {player_name or 'the player'} (Level: {player_level or 'beginner'})

COACHING APPROACH:
- Give direct, actionable tennis advice
- If your complete advice would naturally be 3+ sentences, split into exactly 2 consecutive messages
- Send both messages immediately, one after another
- Message 1: Core concept (technique/explanation) - 1-2 sentences
- Message 2: Application (drill/practice method) + follow-up question - 1-2 sentences
- If you have even more information, end Message 2 with: "Want me to explain more about [specific aspect]?"
- Continue with more detail if player shows any interest (yes/sure/tell me more/questions about the topic)
- Ask 1 specific follow-up question
- End with encouragement like "How does that sound?" or "Ready to try this?"
- Focus on technique, solo drills, or mental game advice
- Be encouraging and supportive
- Remember you are coaching remotely - focus on what they can practice alone

MEMORY RULES:
- NEVER ask about their level - you know they are {player_level or 'a beginner'}
- NEVER ask their name - you are coaching {player_name or 'this player'}
- Remember what you have discussed in this session

{session_context}{recent_conversation}

Player question: "{prompt}"

Provide direct coaching advice:"""

        response = query_claude(claude_client, claude_only_prompt)
        return response, []
    
    # Pinecone modes (Auto or Always)
    else:
        # Query Pinecone
        chunks = query_pinecone(index, prompt, top_k)
        
        # Check relevance for Auto mode
        if coaching_mode == "🤖 Auto (Smart Fallback)":
            fallback_threshold = st.session_state.get('admin_fallback_threshold', 0.45)
            relevant_chunks = [chunk for chunk in chunks if chunk['score'] >= fallback_threshold]
            max_relevance = max([chunk['score'] for chunk in chunks]) if chunks else 0.0
            
            if not relevant_chunks:
                # Fallback to Claude-only
                st.session_state.last_coaching_mode_used = f"⚠️ Fell back to Claude-only (max relevance: {max_relevance:.2f})"
                
                # Use Claude-only logic (same as above)
                recent_conversation = ""
                if len(st.session_state.messages) > 1:
                    recent_conversation = "\nCURRENT SESSION CONVERSATION:\n"
                    for msg in st.session_state.messages[-20:]:
                        role = "Player" if msg['role'] == 'user' else "Coach Taai"
                        recent_conversation += f"{role}: {msg['content']}\n"
                
                session_context = ""
                if coaching_history and len(coaching_history) > 0 and len(st.session_state.messages) <= 4:
                    last_session = coaching_history[0]
                    if last_session.get('technical_focus'):
                        session_context = f"\nPrevious session focus: {last_session['technical_focus']}"
                        session_context += f"\nNOTE: Focus on current conversation topic, not previous session topics."
                
                claude_only_prompt = f"""You are Coach Taai, a professional tennis coach providing remote coaching advice through chat.

{get_coaching_personality_enhancement()}

Player: {player_name or 'the player'} (Level: {player_level or 'beginner'})

COACHING APPROACH:
- Give direct, actionable tennis advice
- Ask 1-2 follow-up questions about their specific situation  
- End with encouragement like "How does that sound?" or "Ready to try this?"
- Keep responses SHORT (1-2 sentences total)
- Focus on technique, solo drills, or mental game advice
- Be encouraging and supportive
- Remember you are coaching remotely - focus on what they can practice alone

MEMORY RULES:
- NEVER ask about their level - you know they are {player_level or 'a beginner'}
- NEVER ask their name - you are coaching {player_name or 'this player'}
- Remember what you have discussed in this session

{session_context}{recent_conversation}

Player question: "{prompt}"

Provide direct coaching advice:"""

                response = query_claude(claude_client, claude_only_prompt)
                return response, []
            
            else:
                # Use relevant chunks
                chunks = relevant_chunks
                st.session_state.last_coaching_mode_used = f"✅ Used Pinecone (relevance: {max_relevance:.2f})"
        
        else:
            # Always use Pinecone mode
            max_relevance = max([chunk['score'] for chunk in chunks]) if chunks else 0.0
            st.session_state.last_coaching_mode_used = f"🔍 Pinecone+Claude forced (relevance: {max_relevance:.2f})"
        
        # Use Pinecone + Claude
        prompt_with_context = build_conversational_prompt_with_history(
            prompt, chunks, st.session_state.messages, coaching_history, player_name, player_level
        )
        response = query_claude(claude_client, prompt_with_context)
        return response, chunks

def extract_name_from_response(user_message: str) -> str:
    """
    Enhanced name extraction - handles complex responses better
    """
    message = user_message.strip()
    
    # Remove common trailing phrases that get captured
    trailing_phrases = [
        ", how are you", ", how are you coach", ", coach", 
        ", how's it going", ", what's up", ", nice to meet you"
    ]
    
    for phrase in trailing_phrases:
        if phrase in message.lower():
            message = message[:message.lower().find(phrase)]
    
    # Handle common response patterns
    if message.lower().startswith(("i'm ", "im ")):
        name = message.split(" ", 1)[1] if len(message.split()) > 1 else message
    elif "i am " in message.lower():
        # Find "i am" anywhere in the message and get the word after it
        parts = message.lower().split("i am ")
        if len(parts) > 1:
            name = parts[1].split()[0] if parts[1].split() else message
        else:
            name = message
    elif "this is " in message.lower():
        # Handle "this is [name]" pattern
        parts = message.lower().split("this is ")
        if len(parts) > 1:
            name = parts[1].split()[0] if parts[1].split() else message
        else:
            name = message
    elif message.lower().startswith(("my name is ", "name is ")):
        name = message.split("is ", 1)[1] if "is " in message else message
    elif message.lower().startswith(("call me ", "it's ", "its ")):
        name = message.split(" ", 1)[1] if len(message.split()) > 1 else message
    else:
        # For simple responses like "Bak" or just a name
        name = message
    
    # Clean up the extracted name
    name = name.strip().strip(',').strip('.')
    
    # Remove any remaining common words
    cleanup_words = ["coach", "tennis", "player", "hi", "hello", "hey", "how", "are", "you"]
    name_words = name.split()
    cleaned_words = [word for word in name_words if word.lower() not in cleanup_words]
    
    if cleaned_words:
        # Take first clean word and capitalize properly
        final_name = cleaned_words[0].title()
        return final_name
    
    # Fallback to first word that's not a common word
    return name.split()[0].title() if name.split() else message.title()

def calculate_days_since_last_session(player_record_id: str) -> int:
    """Calculate days since last session"""
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
            
            for record in records:
                fields = record.get('fields', {})
                record_player_ids = fields.get('player_id', [])
                
                if isinstance(record_player_ids, list) and player_record_id in record_player_ids:
                    last_timestamp = fields.get('timestamp', '')
                    if last_timestamp:
                        try:
                            last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            now_dt = datetime.now(last_dt.tzinfo)
                            days_diff = (now_dt - last_dt).days
                            return days_diff
                        except:
                            pass
                    break
        
        return 7  # Default to 1 week if can't determine
    except Exception as e:
        return 7


def analyze_session_tone(session_summary: dict) -> str:
    """Analyze the tone/mood of the last session"""
    if not session_summary:
        return "neutral"
    
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
    
    # Count indicators
    positive_count = sum(1 for indicator in positive_indicators if indicator in all_text)
    challenging_count = sum(1 for indicator in challenging_indicators if indicator in all_text)
    technical_count = sum(1 for indicator in technical_indicators if indicator in all_text)
    
    # Determine primary tone
    if positive_count >= 2 and positive_count > challenging_count:
        return "positive"
    elif challenging_count >= 2 and challenging_count > positive_count:
        return "challenging"
    elif technical_count >= 2:
        return "technical"
    else:
        return "neutral"


def generate_smart_greeting(player_name: str, days_since: int, session_tone: str, total_sessions: int) -> str:
    """Generate context-aware greeting"""
    
    # Time-based greetings (highest priority)
    if days_since == 0:
        greetings = [
            f"Back already, {player_name}! How's it going?",
            f"Twice in one day, {player_name} - I love the dedication!",
            f"Ready for round two, {player_name}?"
        ]
    elif days_since == 1:
        greetings = [
            f"Back for more, {player_name}! How are you feeling?",
            f"Day two, {player_name}! How's everything feeling?",
            f"Love the commitment, {player_name} - ready to keep working?"
        ]
    elif days_since >= 21:
        greetings = [
            f"{player_name}! Wow, it's been a while - how have you been?",
            f"Hey {player_name}! Great to see you back after so long!",
            f"{player_name}! Good to have you back - how's life been?"
        ]
    elif days_since >= 10:
        greetings = [
            f"{player_name}! Great to have you back!",
            f"Hey {player_name}! It's been a while - how have you been?",
            f"{player_name}! Good to see you again!"
        ]
    else:
        # Recent visits - tone-based greetings
        if session_tone == "positive":
            greetings = [
                f"Hey {player_name}! Still feeling good about that progress?",
                f"{player_name}! How's that confidence been?",
                f"Hi {player_name}! I bet you've been thinking about that breakthrough!"
            ]
        elif session_tone == "challenging":
            greetings = [
                f"Hey {player_name}! How are you feeling today?",
                f"{player_name}! Ready to tackle some tennis?",
                f"Hi {player_name}! How's everything been going?"
            ]
        elif session_tone == "technical":
            greetings = [
                f"Hey {player_name}! How's that technique been working out?",
                f"{player_name}! Have you been practicing what we worked on?",
                f"Hi {player_name}! How's that adjustment feeling?"
            ]
        else:
            # Default based on session frequency
            if total_sessions >= 8:
                greetings = [
                    f"Hey {player_name}! Love seeing you back so consistently!",
                    f"{player_name}! Your dedication is impressive - how are you feeling?",
                    f"Hi {player_name}! Ready for another great session?"
                ]
            elif total_sessions <= 3:
                greetings = [
                    f"Hey {player_name}! Good to see you back!",
                    f"{player_name}! Nice to see you're staying with it!",
                    f"Hi {player_name}! How has tennis been treating you?"
                ]
            else:
                greetings = [
                    f"Hey {player_name}! How's it going?",
                    f"{player_name}! Good to see you again!",
                    f"Hi {player_name}! How have you been?"
                ]
    
    # Get stored recent greetings to avoid repetition
    recent_greetings = st.session_state.get('recent_greetings', [])
    
    # Filter out recently used greetings
    available = [g for g in greetings if g not in recent_greetings]
    
    # If all were used recently, use any available
    if not available:
        available = greetings
    
    # Pick the first available and store it
    selected_greeting = available[0]
    
    # Update recent greetings (keep last 3)
    recent_greetings.append(selected_greeting)
    st.session_state.recent_greetings = recent_greetings[-3:]
    
    return selected_greeting

def generate_followup_message(player_name: str, last_session_summary: dict, session_tone: str) -> str:
    """Generate specific follow-up based on last session priority"""
    
    if not last_session_summary:
        return "What would you like to work on today?"
    
    # Priority 1: Homework/practice check
    homework = last_session_summary.get('homework_assigned', '').strip()
    if homework and len(homework) > 10:  # Meaningful homework content
        if len(homework) > 60:
            homework_preview = homework[:60] + "..."
        else:
            homework_preview = homework
        return f"Did you get a chance to practice what we discussed? {homework_preview} How did it go?"
    
    # Priority 2: Breakthrough follow-up (only if positive tone)
    breakthroughs = last_session_summary.get('key_breakthroughs', '').strip()
    if breakthroughs and len(breakthroughs) > 10 and session_tone == "positive":
        if len(breakthroughs) > 50:
            breakthrough_preview = breakthroughs[:50] + "..."
        else:
            breakthrough_preview = breakthroughs
        return f"How has that breakthrough been working out? {breakthrough_preview}"
    
    # Priority 3: Next session focus
    next_focus = last_session_summary.get('next_session_focus', '').strip()
    if next_focus and len(next_focus) > 10:
        if len(next_focus) > 55:
            focus_preview = next_focus[:55] + "..."
        else:
            focus_preview = next_focus
        return f"Ready to work on what we planned? {focus_preview}"
    
    # Priority 4: Technical follow-up
    technical_focus = last_session_summary.get('technical_focus', '').strip()
    if technical_focus and len(technical_focus) > 10:
        # Look for specific technique mentions
        tech_words = ["forehand", "backhand", "serve", "volley", "grip", "stance", "footwork"]
        mentioned_tech = None
        for tech in tech_words:
            if tech in technical_focus.lower():
                mentioned_tech = tech
                break
        
        if mentioned_tech:
            return f"How has that {mentioned_tech} work been going since last time?"
        else:
            if len(technical_focus) > 45:
                tech_preview = technical_focus[:45] + "..."
            else:
                tech_preview = technical_focus
            return f"How has the work on {tech_preview.lower()} been going?"
    
    # Priority 5: Mental game follow-up
    mental_notes = last_session_summary.get('mental_game_notes', '').strip()
    if mental_notes and len(mental_notes) > 10:
        return f"How has your confidence and mindset been on court?"
    
    # Default fallback
    return "What would you like to focus on today?"

def setup_player_session_with_continuity(player_email: str):
    """
    Enhanced player setup with immediate two-message welcome system and automatic cleanup
    """
    # Clean up any abandoned sessions first (silent cleanup)
    try:
        # Run cleanup silently in background - don't show messages to user
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        
        # Find sessions older than 15 minutes that are still "active"
        from datetime import datetime, timedelta
        cutoff_time = (datetime.now() - timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        params = {
            "filterByFormula": f"AND({{session_status}} = 'active', {{timestamp}} < '{cutoff_time}')",
            "sort[0][field]": "session_id",
            "sort[0][direction]": "desc"
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            all_abandoned_records = response.json().get('records', [])
            
            # Group by session_id and filter out admin sessions
            session_groups = {}
            for record in all_abandoned_records:
                fields = record.get('fields', {})
                session_id = fields.get('session_id')
                message_content = fields.get('message_content', '')
                
                if not session_id or 'hilly spike' in message_content.lower():
                    continue
                
                if session_id not in session_groups:
                    session_groups[session_id] = {
                        'session_id': session_id,
                        'player_ids': fields.get('player_id', []),
                        'message_count': 0
                    }
                session_groups[session_id]['message_count'] += 1
            
            # Only clean up sessions with 4+ messages (real coaching sessions)
            for session_id, session_data in session_groups.items():
                if session_data['message_count'] >= 4 and session_data['player_ids']:
                    player_id = session_data['player_ids'][0]
                    
                    # Mark session as completed
                    session_marked = mark_session_completed(player_id, str(session_id))
                    
                    if session_marked:
                        # Generate summary silently
                        try:
                            # Get connections for summary generation
                            _, claude_client = setup_connections()
                            if claude_client:
                                process_completed_session(player_id, str(session_id), claude_client)
                        except:
                            pass  # Don't let summary errors stop session startup
        
    except Exception as e:
        # Don't let cleanup errors stop the session startup
        pass
    
    # Continue with normal session setup
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
            recent_summaries = get_player_recent_summaries(existing_player['id'], 3)
            st.session_state.coaching_history = recent_summaries
        
        # Generate two-part welcome message
        greeting, followup = enhanced_generate_personalized_welcome_message(
            player_name, 
            session_number, 
            recent_summaries, 
            True
        )
        
        # Store both messages for immediate delivery
        st.session_state.welcome_followup = followup
        
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
            
            # Clear any previous player info
            st.session_state.player_name = ""
            st.session_state.player_level = ""
            st.session_state.welcome_followup = None
            
            return "Hi! I'm Coach Taai, your personal tennis coach. What's your name?"
        else:
            st.error("Error creating player profile. Please try again.")
            return None

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

def analyze_tennis_experience(user_message: str, question_context: str) -> str:
    """
    Use AI to determine player's tennis skill level
    Returns: 'BEGINNER', 'INTERMEDIATE', 'ADVANCED', or 'UNCLEAR'
    """
    try:
        analysis_prompt = f"""
The tennis coach asked: "{question_context}"
Player responded: "{user_message}"

Determine the player's tennis skill level:

BEGINNER: New to tennis, just started, novice, never played, few months experience
INTERMEDIATE: Been playing for a while, has some experience, plays regularly, 1+ years
ADVANCED: Very experienced, competitive play, many years, tournament play
UNCLEAR: Response doesn't clearly indicate skill level

Most players fall into BEGINNER or INTERMEDIATE categories.

Respond with exactly one word: BEGINNER, INTERMEDIATE, ADVANCED, or UNCLEAR
"""

        _, claude_client = setup_connections()
        if claude_client:
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": analysis_prompt}]
            )
            
            analysis = response.content[0].text.strip().upper()
            if analysis in ["BEGINNER", "INTERMEDIATE", "ADVANCED", "UNCLEAR"]:
                return analysis
        
        # Fallback - most players are beginners
        return "BEGINNER"
        
    except Exception as e:
        return "BEGINNER"

def handle_introduction_sequence(user_message: str, claude_client):
    """
    Enhanced introduction sequence with AI-powered experience analysis
    """
    intro_state = st.session_state.get("intro_state", "waiting_for_name")
    
    if intro_state == "waiting_for_name":
        player_name = extract_name_from_response(user_message)
        if player_name:
            st.session_state.collected_name = player_name
            st.session_state.intro_state = "checking_experience"
            return f"Nice to meet you, {player_name}! I'm excited to coach you. Tell me, are you pretty new to tennis?"
    
    elif intro_state == "checking_experience":
        skill_level = analyze_tennis_experience(user_message, "are you pretty new to tennis?")
        
        if skill_level == "BEGINNER":
            success = update_player_info(
                st.session_state.player_record_id,
                st.session_state.collected_name,
                "Beginner"
            )
            st.session_state.intro_completed = True
            st.session_state.intro_state = "complete"
            return "That's wonderful - everyone starts somewhere! What's got you most curious about tennis right now?"
        
        elif skill_level == "INTERMEDIATE":
            success = update_player_info(
                st.session_state.player_record_id,
                st.session_state.collected_name,
                "Intermediate"
            )
            st.session_state.intro_completed = True
            st.session_state.intro_state = "complete"
            return "Great to have an intermediate player! What aspect of your game would you like to work on today?"
        
        elif skill_level == "ADVANCED":
            success = update_player_info(
                st.session_state.player_record_id,
                st.session_state.collected_name,
                "Advanced"
            )
            st.session_state.intro_completed = True
            st.session_state.intro_state = "complete"
            return "Excellent! I love working with advanced players. What specific skills are you looking to refine?"
        
        else:  # UNCLEAR
            st.session_state.intro_state = "asking_time"
            return "Tell me a bit more about your tennis journey - I'd love to understand where you're coming from."
    
    elif intro_state == "asking_time":
        # Use AI to analyze their detailed response
        skill_level = analyze_tennis_experience(user_message, "tell me about your tennis journey")
        
        success = update_player_info(
            st.session_state.player_record_id,
            st.session_state.collected_name,
            skill_level.title()  # Convert to proper case
        )
        
        st.session_state.intro_completed = True
        st.session_state.intro_state = "complete"
        
        if skill_level == "INTERMEDIATE":
            return "I can tell you've put in some good work! What's on your mind for today's session?"
        elif skill_level == "ADVANCED":
            return "Impressive tennis background! What would you like to focus on in our session?"
        else:  # BEGINNER or fallback
            return "Perfect! What would you like to work on together today?"
    
    return None

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

def display_admin_interface(index, claude_client):
    """Enhanced admin interface reading from Active_Sessions for resource analytics"""
    st.title("🔧 Tennis Coach AI - Admin Interface")
    st.markdown("### Session Management & Player Analytics")
    
    # ADMIN COACHING MODE CONTROL
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        coaching_mode = st.radio(
            "🎯 Coaching Mode (Controls All User Sessions)",
            ["🤖 Auto (Smart Fallback)", "🔍 Pinecone + Claude", "🧠 Claude Only"],
            index=0 if not st.session_state.get('admin_coaching_mode') else 
                  ["🤖 Auto (Smart Fallback)", "🔍 Pinecone + Claude", "🧠 Claude Only"].index(st.session_state.get('admin_coaching_mode')),
            help="Auto: Pinecone+Claude, falls back to Claude-only if relevance < 0.65\nPinecone+Claude: Always uses database\nClaude Only: Uses Claude's general knowledge"
        )
        st.session_state.admin_coaching_mode = coaching_mode
    
    with col2:
        if coaching_mode in ["🤖 Auto (Smart Fallback)", "🔍 Pinecone + Claude"]:
            top_k = st.slider("Coaching resources", 1, 8, 3, key="admin_coaching_resources_slider")
            st.session_state.admin_top_k = top_k
        else:
            st.session_state.admin_top_k = 0
    
        # Add fallback threshold slider for Auto mode
        if coaching_mode == "🤖 Auto (Smart Fallback)":
            threshold = st.slider(
                "Fallback Threshold", 
                0.20, 0.80, 0.45, 0.05,
                key="admin_fallback_threshold_slider",
                help="Higher = stricter (more fallbacks), Lower = more permissive (fewer fallbacks)"
            )
            st.session_state.admin_fallback_threshold = threshold
        else:
            st.session_state.admin_fallback_threshold = 0.45  # Default
    
        # Show current mode status
        if 'last_coaching_mode_used' in st.session_state:
            st.markdown("**Last Response Mode:**")
            st.markdown(st.session_state.last_coaching_mode_used)
    
    st.markdown("---")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 All Sessions", "👥 Player Engagement", "🧪 RAG Sandbox", "🔧 Cleanup Test", "📈 Fallback Analysis"])    
    
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
                                        <strong> Player:</strong><br>
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
                                        <strong>Coach Taai:</strong>{resource_indicator}<br>
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
                                                <strong>Player:</strong><br>
                                                {content}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif role == 'coach':
                                        resource_indicator = f" 📚 {resources_used}" if resources_used > 0 else ""
                                        st.markdown(f"""
                                        <div style="display: flex; justify-content: flex-end; margin: 10px 0;">
                                            <div style="background-color: #E8F5E8; padding: 10px 15px; border-radius: 18px; max-width: 70%; border: 1px solid #C8E6C9;">
                                                <strong>Coach Taai:</strong>{resource_indicator}<br>
                                                {content}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        if resources_used > 0 and msg.get('resource_details'):
                                            with st.expander(f"📊 View {resources_used} coaching resources"):
                                                st.text(msg['resource_details'])
                else:
                    st.warning("No sessions found for this player.")
    
    with tab4:
        st.markdown("### 🔧 Session Cleanup Testing")
        st.markdown("Test the abandoned session cleanup function safely.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🧪 Dry Run Test")
            st.markdown("See what would be cleaned up:")
            if st.button("🔍 Run Dry Run Test", type="secondary"):
                with st.spinner("Checking for abandoned sessions..."):
                    cleanup_abandoned_sessions(claude_client, dry_run=True)
        
        with col2:
            st.markdown("#### 📋 Preview & Select")
            st.markdown("Preview sessions and clean up individually:")
            if st.button("👁️ Preview Sessions", type="secondary"):
                with st.spinner("Loading session previews..."):
                    cleanup_abandoned_sessions(claude_client, dry_run=False, preview_mode=True)
        
        with col3:
            st.markdown("#### ⚠️ Bulk Cleanup")
            st.markdown("Clean up all sessions at once:")
            if st.button("🧹 Bulk Cleanup", type="primary"):
                if st.checkbox("I understand this will modify the database"):
                    with st.spinner("Cleaning up all abandoned sessions..."):
                        cleanup_abandoned_sessions(claude_client, dry_run=False)
                else:
                    st.warning("Please check the confirmation box first.")
    
    with tab3:
        try:
            from rag_sandbox import display_rag_sandbox_interface
            display_rag_sandbox_interface(index, claude_client, get_embedding)
        except Exception as e:
            st.error(f"RAG Sandbox error: {e}")
            import traceback
            st.code(traceback.format_exc())

    with tab5:
        st.markdown("### 📈 Fallback Analysis & Content Gap Detection")
        
        # Create subtabs for the two features
        analysis_tab1, analysis_tab2 = st.tabs(["🔍 Individual Session Analysis", "📊 Content Gap Detection"])
        
        with analysis_tab1:
            st.markdown("#### 🔍 Individual Session Analysis")
            st.markdown("Analyze fallback patterns for specific sessions")
            
            # Session selector for detailed analysis
            sessions = get_all_coaching_sessions()
            if sessions:
                session_options = {}
                for session in sessions[:20]:  # Show last 20 sessions
                    timestamp = session['timestamp']
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        formatted_time = dt.strftime("%m/%d %H:%M")
                    except:
                        formatted_time = "Unknown time"
                    
                    status_emoji = "✅" if session['status'] == 'completed' else "🟡"
                    fallback_indicator = "⚠️" if session['total_resources'] == 0 else "🔍"
                    display_name = f"{status_emoji} {fallback_indicator} Session {session['session_id']} | {session['message_count']} msgs | {formatted_time}"
                    session_options[display_name] = session['session_id']
                
                selected_session_display = st.selectbox(
                    "Select session for detailed fallback analysis:",
                    options=list(session_options.keys()),
                    key="fallback_session_selector"
                )
                
                if selected_session_display:
                    selected_session_id = session_options[selected_session_display]
                    
                    # Get detailed fallback analysis
                    with st.spinner("Analyzing session fallback patterns..."):
                        fallback_details = analyze_session_fallback_details(selected_session_id)
                    
                    if fallback_details:
                        st.markdown("##### 📋 Response-by-Response Analysis")
                        
                        # Create analysis table
                        analysis_data = []
                        for detail in fallback_details:
                            analysis_data.append({
                                'Response #': detail['message_order'],
                                'Mode Used': detail['mode_used'],
                                'Details': detail['mode_details'],
                                'Resources': detail['resources_used'],
                                'Preview': detail['message_preview']
                            })
                        
                        if analysis_data:
                            df = pd.DataFrame(analysis_data)
                            st.dataframe(df, use_container_width=True)
                            
                            # Show detailed resource info for responses that used Pinecone
                            st.markdown("##### 🔍 Detailed Resource Analysis")
                            for detail in fallback_details:
                                if detail['resources_used'] > 0 and detail['resource_details']:
                                    with st.expander(f"Response #{detail['message_order']}: {detail['mode_used']} {detail['mode_details']}"):
                                        st.markdown("**Coach Response Preview:**")
                                        st.write(detail['message_preview'])
                                        st.markdown("**Resources Used:**")
                                        st.text(detail['resource_details'])
                                        
                                        # Show relevance scores if available
                                        if detail['relevance_scores']:
                                            st.markdown("**Relevance Scores:**")
                                            for i, score in enumerate(detail['relevance_scores']):
                                                color = "🟢" if score >= 0.7 else "🟡" if score >= 0.5 else "🔴"
                                                st.write(f"{color} Resource {i+1}: {score:.3f}")
                        else:
                            st.info("No coach responses found in this session.")
                    else:
                        st.warning("Could not analyze this session.")
            else:
                st.warning("No sessions available for analysis.")
        
        with analysis_tab2:
            st.markdown("#### 📊 Content Gap Detection")
            st.markdown("Identify topics that frequently trigger fallbacks")
            
            if st.button("🔍 Analyze Content Gaps", type="primary"):
                with st.spinner("Analyzing recent coaching sessions for content gaps..."):
                    gap_analysis = detect_content_gaps()
                
                if gap_analysis:
                    # Overall statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Responses Analyzed", gap_analysis['total_responses'])
                    with col2:
                        st.metric("Fallback Count", gap_analysis['fallback_count'])
                    with col3:
                        fallback_rate = gap_analysis['fallback_rate']
                        st.metric("Fallback Rate", f"{fallback_rate:.1f}%")
                    
                    st.markdown("---")
                    
                    # Content gap insights
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("##### 🔴 Topics Triggering Fallbacks")
                        st.markdown("*These topics need more content in your database*")
                        
                        if gap_analysis['common_fallback_topics']:
                            for keyword, count in gap_analysis['common_fallback_topics']:
                                st.write(f"• **{keyword}** - {count} fallbacks")
                        else:
                            st.info("No common fallback topics found.")
                        
                        # Show recent fallback examples
                        if gap_analysis['recent_fallbacks']:
                            st.markdown("##### 📝 Recent Fallback Examples")
                            for fallback in gap_analysis['recent_fallbacks']:
                                st.write(f"• *\"{fallback['user_query']}\"*")
                                if fallback['keywords']:
                                    st.write(f"  Keywords: {', '.join(fallback['keywords'])}")
                    
                    with col2:
                        st.markdown("##### 🟢 High-Performing Topics")
                        st.markdown("*These topics work well with your current database*")
                        
                        if gap_analysis['high_performing_topics']:
                            for keyword, count in gap_analysis['high_performing_topics']:
                                st.write(f"• **{keyword}** - {count} high-relevance responses")
                        else:
                            st.info("No high-performing topics found.")
                        
                        # Show recent success examples
                        if gap_analysis['recent_successes']:
                            st.markdown("##### ✅ Recent Success Examples")
                            for success in gap_analysis['recent_successes']:
                                st.write(f"• *\"{success['user_query']}\"*")
                                st.write(f"  Relevance: {success['relevance']:.2f}")
                                if success['keywords']:
                                    st.write(f"  Keywords: {', '.join(success['keywords'])}")
                    
                    st.markdown("---")
                    
                    # Recommendations
                    st.markdown("##### 💡 Recommendations")
                    if gap_analysis['fallback_rate'] > 30:
                        st.warning(f"⚠️ High fallback rate ({gap_analysis['fallback_rate']:.1f}%) - Consider adding more content to your database")
                    elif gap_analysis['fallback_rate'] > 15:
                        st.info(f"📊 Moderate fallback rate ({gap_analysis['fallback_rate']:.1f}%) - Room for improvement")
                    else:
                        st.success(f"✅ Low fallback rate ({gap_analysis['fallback_rate']:.1f}%) - Database performing well")
                    
                    # Specific recommendations based on fallback topics
                    if gap_analysis['common_fallback_topics']:
                        st.markdown("**Suggested Content Additions:**")
                        for keyword, count in gap_analysis['common_fallback_topics'][:3]:
                            st.write(f"• Add more **{keyword}** content to reduce {count} fallbacks")
                
                else:
                    st.error("Could not analyze content gaps. Please try again.")
    
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
    
    st.title("🎾 Tennis Coach AI")
    st.markdown("*Your personal tennis coaching assistant*")
    st.markdown("---")
    
    with st.spinner("Connecting to tennis coaching database..."):
        index, claude_client = setup_connections()
    
    if not index or not claude_client:
        st.error("Failed to connect to coaching systems. Please check API keys.")
        st.stop()
    
    # CHECK FOR ADMIN MODE AFTER CONNECTIONS ARE ESTABLISHED
    if st.session_state.get('admin_mode', False):
        display_admin_interface(index, claude_client)
        return
    
    with st.sidebar:
        st.header("🔧 Admin Controls")
    
        # Get coaching mode from session state (set by admin)
        coaching_mode = st.session_state.get('admin_coaching_mode', '🤖 Auto (Smart Fallback)')
        top_k = st.session_state.get('admin_top_k', 3)
    
        if st.button("🔄 New Session"):
            st.session_state.messages = []
            st.session_state.conversation_log = []
            st.session_state.player_setup_complete = False
            st.session_state.welcome_followup = None
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
                        
                        # Add the followup message immediately if it exists
                        followup_msg = st.session_state.get('welcome_followup')
                        if followup_msg:
                            st.session_state.messages.append({"role": "assistant", "content": followup_msg})
                        
                        # Log both messages
                        if st.session_state.get("player_record_id"):
                            # Log greeting message
                            st.session_state.message_counter += 1
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
                            
                            # Log followup message if exists
                            if followup_msg:
                                st.session_state.message_counter += 1
                                log_message_to_sss(
                                    st.session_state.player_record_id,
                                    session_id,
                                    st.session_state.message_counter,
                                    "assistant",
                                    followup_msg
                                )
                                log_message_to_conversation_log(
                                    st.session_state.player_record_id,
                                    session_id,
                                    st.session_state.message_counter,
                                    "assistant",
                                    followup_msg
                                )
                        
                        st.success("Welcome! Ready to start your coaching session.")
                        st.rerun()
        return
    
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
                st.success("🎾 **Session Complete!** Thanks for training with Coach Taai today.")
                if st.button("🔄 Start New Session", type="primary"):
                    for key in list(st.session_state.keys()):
                        if key not in ['player_email', 'player_record_id']:
                            del st.session_state[key]
                    st.rerun()
                return
        
        # SMART COACHING MODE WITH THREE OPTIONS
        with st.chat_message("assistant"):
            with st.spinner("Coach is thinking..."):
                response, chunks = get_smart_coaching_response(
                    prompt, index, claude_client, coaching_mode, top_k
                )
                
                st.markdown(response)
                
                st.session_state.message_counter += 1
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                # DUAL LOGGING: Log coach response with chunks info
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

if __name__ == "__main__":
    main()
