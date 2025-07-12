import streamlit as st
import os
import json
from typing import List, Dict
import time
import pandas as pd
from datetime import datetime, timedelta
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

def build_clean_coaching_prompt(question: str, chunks: List[Dict], conversation_history: List[Dict], player_name: str = "", player_level: str = "") -> str:
    """Clean coaching prompt without debug text"""
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
        history_text = "\nRecent conversation:\n"
        for msg in conversation_history[-6:]:
            role = "Player" if msg['role'] == 'user' else "Coach"
            history_text += f"{role}: {msg['content']}\n"
    
    player_context = f"\nPlayer: {player_name} ({player_level} level)" if player_name and player_level else ""
    
    return f"""You are Coach TA, a professional tennis coach providing remote coaching advice through chat.

Guidelines:
- Keep responses concise and focused (2-3 sentences max for simple questions)
- For complex topics, you may give longer advice but break it naturally
- Focus on ONE specific tip or technique per response
- Give practical advice for solo practice
- Ask engaging follow-up questions
- Be encouraging and supportive
- Avoid suggesting activities requiring a coach present
{player_context}

{history_text}

Professional Coaching Resources:
{context_text}

Current Player Question: "{question}"

Respond naturally as Coach TA:"""

def query_claude(client, prompt: str) -> str:
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

def process_coaching_response(response_text: str) -> tuple:
    """
    Process Claude's response and break into natural parts if needed
    Returns: (first_message, second_message, should_delay)
    """
    response_text = response_text.strip()
    
    # Check if response should be broken into two parts
    word_count = len(response_text.split())
    
    if word_count > 45:  # If response is long, try to break it naturally
        sentences = response_text.split('. ')
        
        if len(sentences) >= 3:
            # Find a good breaking point around middle
            mid_point = len(sentences) // 2
            
            # Look for natural break indicators
            for i in range(mid_point - 1, min(len(sentences), mid_point + 2)):
                sentence = sentences[i].lower()
                
                # Break before questions or practice suggestions
                if any(word in sentence for word in ['?', 'how', 'what', 'try', 'practice', 'work on']):
                    first_part = '. '.join(sentences[:i])
                    second_part = '. '.join(sentences[i:])
                    
                    # Add periods if missing
                    if first_part and not first_part.endswith('.'):
                        first_part += '.'
                    if second_part and not second_part.endswith('.') and not second_part.endswith('?'):
                        second_part += '.'
                    
                    return (first_part, second_part, True)
        
        # If no good breaking point found, break at sentence boundary near middle
        if len(sentences) >= 2:
            mid_sentence = len(sentences) // 2
            first_part = '. '.join(sentences[:mid_sentence])
            second_part = '. '.join(sentences[mid_sentence:])
            
            # Add periods if missing
            if first_part and not first_part.endswith('.'):
                first_part += '.'
            if second_part and not second_part.endswith('.') and not second_part.endswith('?'):
                second_part += '.'
            
            return (first_part, second_part, True)
    
    # Return as single message if short or couldn't break naturally
    return (response_text, None, False)

def find_player_by_email(email: str):
    try:
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

def create_new_player(email: str, name: str = "", tennis_level: str = ""):
    try:
        email = email.lower().strip()
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        fields = {
            "email": email,
            "name": name if name else "",
            "primary_goals": [],
            "personality_notes": "",
            "total_sessions": 1,
            "first_session_date": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "player_status": "Active"
        }
        
        if tennis_level and tennis_level in ["Beginner", "Intermediate", "Advanced"]:
            fields["tennis_level"] = tennis_level
        
        data = {"fields": fields}
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None

def update_player_info(player_id: str, name: str = "", tennis_level: str = ""):
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_id}"
        headers = {
            "Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}",
            "Content-Type": "application/json"
        }
        
        update_data = {"fields": {}}
        if name:
            update_data["fields"]["name"] = name
        if tennis_level:
            update_data["fields"]["tennis_level"] = tennis_level
        
        response = requests.patch(url, headers=headers, json=update_data)
        return response.status_code == 200
    except Exception as e:
        return False

def get_player_recent_summaries(player_record_id: str, limit: int = 3) -> list:
    try:
        url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Session_Summaries"
        headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
        params = {
            "sort[0][field]": "session_number", 
            "sort[0][direction]": "desc",
            "maxRecords": 50
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            all_records = response.json().get('records', [])
            
            matching_summaries = []
            for record in all_records:
                fields = record.get('fields', {})
                player_ids = fields.get('player_id', [])
                
                if isinstance(player_ids, list) and player_record_id in player_ids:
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

def calculate_days_since_last_session(player_record_id: str) -> int:
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

def generate_smart_greeting(player_name: str, player_record_id: str, last_session_summary: dict, total_sessions: int) -> str:
    """Generate intelligent, varied greeting based on player history"""
    days_since = calculate_days_since_last_session(player_record_id)
    
    # Get recent greetings to avoid repetition
    used_greetings = st.session_state.get('recent_greetings', [])
    
    # Time-based greetings (highest priority)
    if days_since == 0:
        greetings = [
            "Back already! How's it going?",
            "Twice in one day - I love the dedication!",
            "Ready for round two?"
        ]
    elif days_since == 1:
        greetings = [
            "Back for more! How are you feeling?",
            "Love the commitment - ready to keep working?",
            "Day two! How's everything feeling?"
        ]
    elif days_since >= 22:
        greetings = [
            f"{player_name}! Wow, it's been a while - how have you been?",
            f"Hey {player_name}! Great to see you back after so long!",
            f"{player_name}! Good to have you back - how's life been?"
        ]
    elif days_since >= 11:
        greetings = [
            f"{player_name}! Great to have you back!",
            f"Hey {player_name}! It's been a while - how have you been?",
            f"{player_name}! Good to see you again!"
        ]
    else:
        # Recent visit - check session tone
        session_tone = analyze_session_tone(last_session_summary)
        
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
            # Default friendly greetings
            greetings = [
                f"Hey {player_name}! How's it going?",
                f"{player_name}! Good to see you again!",
                f"Hi {player_name}! How have you been?"
            ]
    
    # Filter out recently used greetings
    available = [g for g in greetings if g not in used_greetings]
    if not available:
        available = greetings
    
    # Store in session memory (keep last 3)
    greeting = available[0]
    recent_greetings = st.session_state.get('recent_greetings', [])
    recent_greetings.append(greeting)
    st.session_state.recent_greetings = recent_greetings[-3:]
    
    return greeting

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
    
    positive_indicators = [
        'breakthrough', 'progress', 'improvement', 'great', 'excellent', 'good',
        'clicked', 'got it', 'makes sense', 'comfortable', 'confident'
    ]
    
    challenging_indicators = [
        'struggle', 'difficult', 'frustrating', 'hard time', 'trouble',
        'inconsistent', 'issues', 'problems', 'challenging'
    ]
    
    technical_indicators = [
        'grip', 'stance', 'follow-through', 'technique', 'mechanics',
        'form', 'adjustment', 'forehand', 'backhand', 'serve'
    ]
    
    positive_count = sum(1 for indicator in positive_indicators if indicator in all_text)
    challenging_count = sum(1 for indicator in challenging_indicators if indicator in all_text)
    technical_count = sum(1 for indicator in technical_indicators if indicator in all_text)
    
    if positive_count >= 2 and positive_count > challenging_count:
        return "positive"
    elif challenging_count >= 2 and challenging_count > positive_count:
        return "challenging"
    elif technical_count >= 2:
        return "technical"
    else:
        return "neutral"

def generate_followup_message(player_name: str, last_session_summary: dict) -> str:
    """Generate the second message based on what happened last session"""
    if not last_session_summary:
        return "What's on your mind for today's session?"
    
    # Priority 1: Homework/practice check
    homework = last_session_summary.get('homework_assigned', '').strip()
    if homework:
        homework_preview = homework[:75] + "..." if len(homework) > 75 else homework
        return f"Did you get a chance to practice what we discussed? {homework_preview} How did it go?"
    
    # Priority 2: Technical follow-up
    technical_focus = last_session_summary.get('technical_focus', '').strip()
    if technical_focus:
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
    
    # Priority 3: Breakthrough follow-up
    breakthroughs = last_session_summary.get('key_breakthroughs', '').strip()
    if breakthroughs:
        breakthrough_preview = breakthroughs[:60] + "..." if len(breakthroughs) > 60 else breakthroughs
        return f"How has that breakthrough been working out? {breakthrough_preview}"
    
    # Default fallback
    return "What would you like to focus on today?"

def setup_player_session_with_continuity(player_email: str):
    """Enhanced player setup with smart two-message welcome system"""
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
        
        # Generate smart greeting
        if recent_summaries:
            greeting = generate_smart_greeting(player_name, existing_player['id'], recent_summaries[0], session_number)
            followup = generate_followup_message(player_name, recent_summaries[0])
            
            # Store followup for 5-second timer
            st.session_state.pending_followup = followup
            st.session_state.followup_timer_start = time.time()
            st.session_state.followup_sent = False
        else:
            greeting = f"Hey {player_name}! Coach TA here. Session #{session_number} 🎾 What would you like to work on today?"
        
        # Update session count
        update_player_session_count(existing_player['id'])
        
        # Store player info
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
            
            # Set introduction state
            st.session_state.intro_state = "waiting_for_name"
            st.session_state.intro_completed = False
            
            # Clear player info and pending messages
            st.session_state.player_name = ""
            st.session_state.player_level = ""
            st.session_state.pending_followup = None
            st.session_state.followup_sent = False
            
            return "Hi! I'm Coach TA, your personal tennis coach. What's your name?"
        else:
            st.error("Error creating player profile. Please try again.")
            return None

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

def check_and_send_followup_message():
    """Check if it's time to send the followup message (5-second timer)"""
    if not st.session_state.get('pending_followup') or st.session_state.get('followup_sent'):
        return False, None
    
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
    """Handle when user responds before the 5-second timer - COMPLETELY DISABLE TIMER"""
    if st.session_state.get('pending_followup') and not st.session_state.get('followup_sent'):
        followup_msg = st.session_state.pending_followup
        
        # CRITICAL: Mark as sent AND clear all timer state to prevent automatic sending
        st.session_state.followup_sent = True
        st.session_state.pending_followup = None
        st.session_state.followup_timer_start = 0  # Clear timer completely
        
        return True, followup_msg
    
    return False, None

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
    """Enhanced logging that includes resource relevance data"""
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

def extract_name_from_response(user_message: str) -> str:
    """Extract player name from their response with better cleanup"""
    message = user_message.strip()
    
    def clean_name(raw_name):
        """Clean extracted name by removing everything after punctuation"""
        # Remove everything after comma, period, question mark, exclamation, etc.
        for punct in [',', '.', '?', '!', ';', ':', ' and ', ' but ', ' how ', ' what ', ' where ']:
            if punct in raw_name:
                raw_name = raw_name.split(punct)[0]
        return raw_name.strip().title()
    
    # Handle common response patterns
    if message.lower().startswith(("i'm ", "im ", "i am ")):
        raw_name = message.split(" ", 2)[2] if len(message.split()) > 2 else message.split(" ", 1)[1]
        return clean_name(raw_name)
    
    elif message.lower().startswith(("my name is ", "name is ")):
        raw_name = message.split("is ", 1)[1]
        return clean_name(raw_name)
    
    elif message.lower().startswith(("call me ", "it's ", "its ")):
        raw_name = message.split(" ", 1)[1]
        return clean_name(raw_name)
    
    else:
        # Assume the whole message is the name, but clean it
        return clean_name(message)

def assess_player_level_from_conversation(conversation_history: list, claude_client) -> str:
    """Simple conversational assessment - when in doubt, default to Beginner"""
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
        has_lessons = any(indicator in all_responses for indicator in lesson_indicators)
        no_lessons = any(indicator in all_responses for indicator in no_lesson_indicators)
        
        # Decision logic for 1+ year players
        if has_regular_play and has_lessons:
            return "Intermediate"
        elif has_regular_play and not no_lessons:  # Regular play, lessons unclear
            return "Intermediate"
        elif has_lessons:  # Has lessons
            return "Intermediate"
        elif no_lessons:  # No lessons
            return "Beginner"
        else:
            # When in doubt for 1+ year players, lean toward Intermediate
            return "Intermediate"
    
    # DEFAULT: When in doubt, return Beginner
    return "Beginner"

def handle_introduction_sequence(user_message: str, claude_client):
    """Enhanced introduction sequence with conversational level assessment"""
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
        
        # Store player info in session state
        st.session_state.player_name = st.session_state.collected_name
        st.session_state.player_level = assessed_level
        
        # Acknowledge their level naturally
        if assessed_level == "Intermediate":
            return "Sounds like you've got some good experience! What's on your mind for today's session?"
        else:
            return "Perfect! What would you like to work on today?"
    
    return None

def detect_session_end(message_content: str, conversation_history: list = None) -> dict:
    """Intelligent session end detection with context awareness"""
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
    
    # Check definitive endings
    for phrase in definitive_endings:
        if phrase in message_lower:
            return {'should_end': True, 'confidence': 'high', 'needs_confirmation': False}
    
    # Check likely endings (coaching-specific)
    for phrase in likely_endings:
        if phrase in message_lower:
            return {'should_end': True, 'confidence': 'medium', 'needs_confirmation': True}
    
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

def generate_dynamic_session_ending(conversation_history: list, player_name: str = "") -> str:
    """Generate personalized session ending message"""
    import random
    
    # Analyze the session to personalize the message
    session_content = " ".join([msg['content'].lower() for msg in conversation_history if msg['role'] == 'user'])
    
    # Detect what they worked on
    techniques = []
    if any(word in session_content for word in ['forehand']):
        techniques.append('forehand')
    if any(word in session_content for word in ['backhand']):
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
        "Your dedication to getting better really shows!"
    ]
    
    # Learning/challenge acknowledgments
    if techniques:
        technique_work = techniques[0] if len(techniques) == 1 else f"{techniques[0]} and {techniques[1]}"
        learning_phrases = [
            f"Working on {technique_work} takes patience - you're on the right track!",
            f"Those {technique_work} adjustments we discussed will click with practice!",
            f"The {technique_work} work we covered today will pay off on court!"
        ]
    else:
        learning_phrases = [
            "The concepts we covered today will make more sense as you practice them!",
            "Breaking down technique like this is how real improvement happens!",
            "Those adjustments take time to feel natural - trust the process!"
        ]
    
    # Motivational closings
    motivation_phrases = [
        "Keep that curiosity and drive - it's your biggest asset! 🎾",
        "You've got the right mindset to take your game to the next level! 🎾",
        "Stay patient with yourself and trust the process - you're improving! 🎾",
        "Keep asking great questions and putting in the work - exciting progress ahead! 🎾"
    ]
    
    # Combine randomly
    effort = random.choice(effort_phrases)
    learning = random.choice(learning_phrases)
    motivation = random.choice(motivation_phrases)
    
    return f"{effort} {learning} {motivation}"

def is_valid_email(email: str) -> bool:
    """Robust email validation using regex pattern"""
    import re
    
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

def main():
    st.set_page_config(
        page_title="Tennis Coach AI",
        page_icon="🎾",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # CHECK FOR ADMIN MODE FIRST (you'll need to add your admin functions here)
    if st.session_state.get('admin_mode', False):
        # display_admin_interface()  # Add your admin interface function
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
        st.header("🔧 Settings")
        top_k = st.slider("Coaching resources", 1, 8, 3)
        
        if st.button("🔄 New Session"):
            st.session_state.messages = []
            st.session_state.conversation_log = []
            st.session_state.player_setup_complete = False
            # Clear message breaking state
            st.session_state.pending_second_message = None
            st.session_state.second_message_chunks = None
            st.session_state.second_message_timer = None
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
    should_send_followup, followup_msg = check_and_send_followup_message()
    if should_send_followup and followup_msg:
        with st.chat_message("assistant"):
            st.markdown(followup_msg)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": followup_msg
        })
        
        # Log followup message
        if st.session_state.get("player_record_id"):
            st.session_state.message_counter += 1
            log_message_to_sss(
                st.session_state.player_record_id,
                st.session_state.session_id,
                st.session_state.message_counter,
                "assistant",
                followup_msg
            )
            log_message_to_conversation_log(
                st.session_state.player_record_id,
                st.session_state.session_id,
                st.session_state.message_counter,
                "assistant",
                followup_msg
            )
        
        st.rerun()
    
    # CHECK FOR PENDING SECOND MESSAGE (1.5-second timer)
    if st.session_state.get('pending_second_message'):
        start_time = st.session_state.get('second_message_timer', 0)
        current_time = time.time()
        
        if current_time - start_time >= 1.5:  # 1.5 seconds delay
            second_msg = st.session_state.pending_second_message
            chunks = st.session_state.get('second_message_chunks', [])
            
            # Send second message
            with st.chat_message("assistant"):
                st.markdown(second_msg)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": second_msg
            })
            
            # Log second message
            if st.session_state.get("player_record_id"):
                st.session_state.message_counter += 1
                log_message_to_sss(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    second_msg,
                    chunks
                )
                log_message_to_conversation_log(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    second_msg,
                    chunks
                )
            
            # Clear pending message
            st.session_state.pending_second_message = None
            st.session_state.second_message_chunks = None
            st.session_state.second_message_timer = None
            
            st.rerun()  # Refresh to show the second message
    
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
        
        # CRITICAL FIX: Handle followup interruption FIRST, before processing user input
        should_send_interrupt_followup, interrupt_followup = handle_user_response_during_timer()
        
        # Add user message to display immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # If there was a pending followup, send it now
        if should_send_interrupt_followup and interrupt_followup:
            with st.chat_message("assistant"):
                st.markdown(interrupt_followup)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": interrupt_followup
            })
            
            # Log interrupt followup
            if st.session_state.get("player_record_id"):
                st.session_state.message_counter += 1
                log_message_to_sss(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    interrupt_followup
                )
                log_message_to_conversation_log(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    interrupt_followup
                )
        
        # Smart session end detection
        end_result = detect_session_end(prompt, st.session_state.messages)
        
        if end_result['should_end']:
            if end_result['needs_confirmation']:
                st.session_state.pending_session_end = True
                st.session_state.end_confidence = end_result['confidence']
            else:
                st.session_state.session_ending = True
        
        # Handle confirmation responses
        if st.session_state.get("pending_session_end") and prompt.lower().strip() in ["yes", "y", "yeah", "yep", "sure"]:
            st.session_state.session_ending = True
            st.session_state.pending_session_end = False
        elif st.session_state.get("pending_session_end") and prompt.lower().strip() in ["no", "n", "nope", "not yet", "continue"]:
            st.session_state.pending_session_end = False
        
        st.session_state.message_counter += 1
        
        # Log user message
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
        
        # Handle introduction sequence for new players
        if not st.session_state.get("intro_completed", True):
            intro_response = handle_introduction_sequence(prompt, claude_client)
            if intro_response:
                with st.chat_message("assistant"):
                    st.markdown(intro_response)
                
                st.session_state.message_counter += 1
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": intro_response
                })
                
                # Log intro response
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
                return
        
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
                log_message_to_conversation_log(
                    st.session_state.player_record_id,
                    st.session_state.session_id,
                    st.session_state.message_counter,
                    "assistant",
                    confirmation_msg
                )
            return
        
        # If session is ending, provide closing response
        if st.session_state.get("session_ending"):
            with st.chat_message("assistant"):
                player_name = st.session_state.get("player_name", "")
                closing_response = generate_dynamic_session_ending(st.session_state.messages, player_name)
                st.markdown(closing_response)
                
                st.session_state.message_counter += 1
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": closing_response
                })
                
                # Log closing response
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
                
                # Show session end message
                st.success("🎾 **Session Complete!** Thanks for training with Coach TA today.")
                if st.button("🔄 Start New Session", type="primary"):
                    for key in list(st.session_state.keys()):
                        if key not in ['player_email', 'player_record_id']:
                            del st.session_state[key]
                    st.rerun()
                return
        
        # Normal message processing (not ending) - WITH MESSAGE BREAKING
        with st.chat_message("assistant"):
            with st.spinner("Coach is thinking..."):
                chunks = query_pinecone(index, prompt, top_k)
                
                if chunks:
                    coaching_history = st.session_state.get('coaching_history', [])
                    
                    # Get current player info
                    player_name = st.session_state.get("player_name", "")
                    player_level = st.session_state.get("player_level", "")
                    
                    full_prompt = build_clean_coaching_prompt(
                        prompt, 
                        chunks, 
                        st.session_state.messages[:-1],
                        player_name,
                        player_level
                    )
                    
                    response = query_claude(claude_client, full_prompt)
                    
                    # Process response for natural breaking
                    first_msg, second_msg, should_delay = process_coaching_response(response)
                    
                    # Send first message
                    st.markdown(first_msg)
                    st.session_state.message_counter += 1
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": first_msg
                    })
                    
                    # Log first message
                    if st.session_state.get("player_record_id"):
                        log_message_to_sss(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
                            st.session_state.message_counter,
                            "assistant",
                            first_msg,
                            chunks
                        )
                        log_message_to_conversation_log(
                            st.session_state.player_record_id,
                            st.session_state.session_id,
                            st.session_state.message_counter,
                            "assistant",
                            first_msg,
                            chunks
                        )
                    
                    # If there's a second message, set it up for delayed sending
                    if second_msg:
                        st.session_state.pending_second_message = second_msg
                        st.session_state.second_message_chunks = chunks
                        st.session_state.second_message_timer = time.time()
                    
                else:
                    error_msg = "Could you rephrase that? I want to give you the best coaching advice possible."
                    st.markdown(error_msg)
                    st.session_state.message_counter += 1
                    
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
                    # Log error message
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
