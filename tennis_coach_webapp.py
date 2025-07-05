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
st.error(f”Missing package: {e}”)
st.stop()

@st.cache_resource
def setup_connections():
try:
pc = Pinecone(api_key=st.secrets[“PINECONE_API_KEY”])
index = pc.Index(st.secrets[“PINECONE_INDEX_NAME”])
claude_client = anthropic.Anthropic(api_key=st.secrets[“ANTHROPIC_API_KEY”])
return index, claude_client
except Exception as e:
st.error(f”Connection error: {e}”)
return None, None

def get_embedding(text: str) -> List[float]:
try:
api_key = st.secrets[“OPENAI_API_KEY”]

```
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
except Exception as e:
    st.error(f"Embedding error: {e}")
    return []
```

def extract_array_value(metadata_field):
if not metadata_field:
return “Not specified”
if isinstance(metadata_field, list):
if len(metadata_field) > 0:
for item in metadata_field:
if item and str(item).strip():
return str(item).strip()
return “Not specified”
if isinstance(metadata_field, str):
if metadata_field.startswith(’[’) and metadata_field.endswith(’]’):
cleaned = metadata_field.strip(’[]’).replace(’”’, ‘’).replace(”’”, “”)
cleaned = ’ ’.join(cleaned.split())
return cleaned if cleaned else “Not specified”
return str(metadata_field).strip() if metadata_field else “Not specified”

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
‘text’: match.metadata.get(‘text_preview’, ‘’),
‘score’: match.score,
‘source’: match.metadata.get(‘source_url’, ‘Unknown’),
‘topics’: match.metadata.get(‘tennis_topics’, ‘’),
‘skill_level’: extract_array_value(match.metadata.get(‘skill_level’)),
‘coaching_style’: extract_array_value(match.metadata.get(‘coaching_style’))
}
for match in results.matches
]
return chunks
except Exception as e:
st.error(f”Pinecone query error: {e}”)
return []

def build_conversational_prompt(question: str, chunks: List[Dict], conversation_history: List[Dict]) -> str:
context_sections = []
for i, chunk in enumerate(chunks):
context_sections.append(f”””
Resource {i+1}:
Topics: {chunk[‘topics’]}
Level: {chunk[‘skill_level’]}
Style: {chunk[‘coaching_style’]}
Content: {chunk[‘text’]}
“””)
context_text = “\n”.join(context_sections)
history_text = “”
if conversation_history:
history_text = “\nPrevious conversation:\n”
for msg in conversation_history[-6:]:
role = “Player” if msg[‘role’] == ‘user’ else “Coach”
history_text += f”{role}: {msg[‘content’]}\n”
return f””“You are a professional tennis coach providing REMOTE coaching advice through chat. The player is not physically with you, so focus on guidance they can apply on their own.

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

Current Player Question: “{question}”

Respond as their remote tennis coach with a SHORT, focused response:”””

def query_claude(client, prompt: str) -> str:
try:
response = client.messages.create(
model=“claude-3-5-sonnet-20241022”,
max_tokens=300,
messages=[
{“role”: “user”, “content”: prompt}
]
)
return response.content[0].text
except Exception as e:
return f”Error generating coaching response: {e}”

def create_session_record(session_id: str, tester_name: str) -> str:
try:
url = f”https://api.airtable.com/v0/{st.secrets[‘AIRTABLE_BASE_ID’]}/Test_Sessions”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”,
“Content-Type”: “application/json”
}
device_info = f”{platform.system()} - {platform.processor()}”
data = {
“fields”: {
“session_id”: session_id,
“tester_name”: tester_name,
“total_messages”: 0,
“device_info”: device_info
}
}
response = requests.post(url, headers=headers, json=data)
if response.status_code == 200:
record_data = response.json()
record_id = record_data[‘id’]
return record_id
else:
return None
except Exception as e:
return None

def log_message(session_record_id: str, message_order: int, role: str, content: str, chunks=None) -> bool:
try:
url = f”https://api.airtable.com/v0/{st.secrets[‘AIRTABLE_BASE_ID’]}/Conversation_Log”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”,
“Content-Type”: “application/json”
}
resource_details = “”
resources_count = 0
if chunks:
resources_count = len(chunks)
resource_details = “\n”.join([
f”Resource {i+1}: {chunk[‘topics’]} (Score: {chunk[‘score’]:.3f}) - {chunk[‘skill_level’]}”
for i, chunk in enumerate(chunks)
])
data = {
“fields”: {
“session_id”: [session_record_id],
“message_order”: message_order,
“role”: role,
“message_content”: content[:100000],
“coaching_resources_used”: resources_count,
“resource_details”: resource_details
}
}
response = requests.post(url, headers=headers, json=data)
return response.status_code == 200
except Exception as e:
return False

def update_session_stats(session_id: str, total_messages: int) -> bool:
try:
url = f”https://api.airtable.com/v0/{st.secrets[‘AIRTABLE_BASE_ID’]}/Test_Sessions”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”
}
params = {
“filterByFormula”: f”{{session_id}} = ‘{session_id}’”
}
response = requests.get(url, headers=headers, params=params)
if response.status_code == 200:
records = response.json().get(‘records’, [])
if records:
record_id = records[0][‘id’]
update_url = f”{url}/{record_id}”
update_data = {
“fields”: {
“total_messages”: total_messages,
“end_time”: time.strftime(”%Y-%m-%dT%H:%M:%S.000Z”)
}
}
update_response = requests.patch(update_url, headers=headers, json=update_data)
return update_response.status_code == 200
return False
except Exception as e:
return False

def find_player_by_email(email: str):
“”“Look up player in SSS Players table by email”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players”
headers = {“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”}
params = {“filterByFormula”: f”{{email}} = ‘{email}’”}

```
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        records = response.json().get('records', [])
        return records[0] if records else None
    return None
except Exception as e:
    return None
```

def create_new_player(email: str):
“”“Create new player record with just email (coach will gather other info)”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”,
“Content-Type”: “application/json”
}

```
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
    
    # TEMPORARY DEBUG - THESE LINES WILL SHOW THE EXACT ERROR:
    st.error(f"Airtable Response Code: {response.status_code}")
    st.error(f"Airtable Response: {response.text}")
    
    if response.status_code == 200:
        return response.json()
    return None
except Exception as e:
    st.error(f"Exception details: {str(e)}")
    return None
```

def update_player_session_count(player_record_id: str):
“”“Update player’s total sessions”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”,
“Content-Type”: “application/json”
}

```
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
```

def detect_session_end(message_content: str) -> bool:
“”“Detect if user message indicates session should end”””
goodbye_phrases = [
“thanks”, “thank you”, “bye”, “goodbye”, “see you”, “done”,
“that’s all”, “finished”, “end session”, “stop”, “quit”,
“done for today”, “good session”, “catch you later”, “later”,
“gotta go”, “have to go”, “thanks coach”, “thank you coach”
]

```
message_lower = message_content.lower().strip()

for phrase in goodbye_phrases:
    if phrase in message_lower:
        return True

if len(message_lower.split()) <= 3:
    ending_words = ["thanks", "bye", "done", "good", "great"]
    if any(word in message_lower for word in ending_words):
        return True

return False
```

def mark_session_completed(player_record_id: str, session_id: str) -> bool:
“”“Mark all active messages for this session as completed”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions”
headers = {“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”}

```
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
```

def get_session_messages(player_record_id: str, session_id: str) -> list:
“”“Retrieve all messages from a completed session”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions”
headers = {“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”}

```
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
```

def generate_session_summary(messages: list, claude_client) -> dict:
“”“Use Claude to generate structured session summary”””
try:
conversation_text = “”
for msg in messages:
role_label = “Player” if msg[‘role’] == ‘player’ else “Coach”
conversation_text += f”{role_label}: {msg[‘content’]}\n\n”

```
    summary_prompt = f"""Analyze this tennis coaching session and extract key information. The session is between a coach and player working on tennis improvement.
```

CONVERSATION:
{conversation_text}

Please analyze and provide a structured summary with these exact sections:

TECHNICAL_FOCUS: What specific tennis techniques were discussed or worked on? (e.g., forehand grip, serve motion, backhand slice)

MENTAL_GAME: Any mindset, confidence, or mental approach topics covered? (e.g., staying calm, visualization, match preparation)

HOMEWORK_ASSIGNED: What practice tasks or exercises were given to the player? (e.g., wall hitting, shadow swings, specific drills)

NEXT_SESSION_FOCUS: Based on this session, what should be the priority for the next coaching session?

KEY_BREAKTHROUGHS: Any important progress moments, “aha” moments, or skill improvements noted?

CONDENSED_SUMMARY: Write a concise 200-300 token summary capturing the essence of this coaching session, focusing on what was learned and accomplished.

Format your response exactly like this:
TECHNICAL_FOCUS: [your analysis]
MENTAL_GAME: [your analysis]  
HOMEWORK_ASSIGNED: [your analysis]
NEXT_SESSION_FOCUS: [your analysis]
KEY_BREAKTHROUGHS: [your analysis]
CONDENSED_SUMMARY: [your analysis]”””

```
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=800,
        messages=[{"role": "user", "content": summary_prompt}]
    )
    
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
    return {
        'technical_focus': 'Summary generation failed',
        'mental_game_notes': '',
        'homework_assigned': '',
        'next_session_focus': 'Continue working on tennis fundamentals',
        'key_breakthroughs': '',
        'condensed_summary': 'Coaching session completed but summary generation encountered an error.'
    }
```

def save_session_summary(player_record_id: str, session_number: int, summary_data: dict, original_message_count: int) -> bool:
“”“Save the generated summary to Session_Summaries table”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Session_Summaries”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”,
“Content-Type”: “application/json”
}

```
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
    
    # TEMPORARY DEBUG FOR SUMMARY SAVING:
    st.error(f"Summary Save Response Code: {response.status_code}")
    st.error(f"Summary Save Response: {response.text}")
    st.error(f"Data being sent: {data}")
    
    return response.status_code == 200
    
except Exception as e:
    st.error(f"Summary save exception: {str(e)}")
    return False
```

def process_completed_session(player_record_id: str, session_id: str, claude_client) -> bool:
“”“Complete session processing: generate summary and save”””
try:
st.error(f”DEBUG: Starting session processing for player {player_record_id}, session {session_id}”)

```
    messages = get_session_messages(player_record_id, session_id)
    st.error(f"DEBUG: Retrieved {len(messages)} messages")
    
    if not messages:
        st.error("DEBUG: No messages found!")
        return False
    
    st.error(f"DEBUG: Generating summary with Claude...")
    summary_data = generate_session_summary(messages, claude_client)
    st.error(f"DEBUG: Summary generated: {list(summary_data.keys())}")
    
    player_url = f"https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Players/{player_record_id}"
    headers = {"Authorization": f"Bearer {st.secrets['AIRTABLE_API_KEY']}"}
    
    player_response = requests.get(player_url, headers=headers)
    if player_response.status_code == 200:
        player_data = player_response.json()
        session_number = player_data.get('fields', {}).get('total_sessions', 1)
        st.error(f"DEBUG: Session number: {session_number}")
    else:
        session_number = 1
        st.error(f"DEBUG: Using default session number: 1")
    
    st.error(f"DEBUG: About to save summary...")
    summary_saved = save_session_summary(
        player_record_id, 
        session_number, 
        summary_data, 
        len(messages)
    )
    
    st.error(f"DEBUG: Summary saved result: {summary_saved}")
    return summary_saved
    
except Exception as e:
    st.error(f"DEBUG: Exception in process_completed_session: {str(e)}")
    return False
```

def show_session_end_message():
“”“Display session completion message”””
st.success(“🎾 **Session Complete!** Thanks for training with Coach TA today.”)
st.info(“💡 **Your session has been saved.** When you return, I’ll remember what we worked on and continue building on your progress!”)

```
if st.button("🔄 Start New Session", type="primary"):
    for key in list(st.session_state.keys()):
        if key not in ['player_email', 'player_record_id']:
            del st.session_state[key]
    st.rerun()
```

def log_message_to_sss(player_record_id: str, session_id: str, message_order: int, role: str, content: str, chunks=None) -> bool:
“”“Log message to SSS Active_Sessions table”””
try:
url = f”https://api.airtable.com/v0/appTCnWCPKMYPUXK0/Active_Sessions”
headers = {
“Authorization”: f”Bearer {st.secrets[‘AIRTABLE_API_KEY’]}”,
“Content-Type”: “application/json”
}

```
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
```

def main():
st.set_page_config(
page_title=“Tennis Coach AI”,
page_icon=“🎾”,
layout=“centered”,
initial_sidebar_state=“collapsed”
)

```
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
                        st.session_state.previous_sessions = player_data.get('total_sessions', 0)
                        player_name = player_data.get('name', 'there')
                        
                        update_player_session_count(existing_player['id'])
                        
                        welcome_type = "returning"
                        session_info = f"This is session #{player_data.get('total_sessions', 0) + 1}"
                        
                    else:
                        new_player = create_new_player(player_email)
                        if new_player:
                            st.session_state.player_record_id = new_player['id']
                            st.session_state.is_returning_player = False
                            st.session_state.previous_sessions = 0
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
                    st.session_state.airtable_record_id = None
                    st.session_state.messages = []
                    st.session_state.conversation_log = []
                    st.session_state.message_counter = 0
                    
                    airtable_record_id = create_session_record(session_id, player_email)
                    if airtable_record_id:
                        st.session_state.airtable_record_id = airtable_record_id
                    
                    if welcome_type == "returning":
                        welcome_msg = f"""👋 Hi! This is your Coach TA. Great to see you back, {player_name}!
```

{session_info} - What shall we work on today?”””
else:
welcome_msg = f””“👋 Hi! This is your Coach TA. {session_info}

I’m here to help you improve your tennis game. What shall we work on today?

I can help with technique, strategy, mental game, or any specific issues you’re having on court.”””

```
                    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
                    st.session_state.conversation_log = [{
                        "role": "assistant", 
                        "content": welcome_msg,
                        "timestamp": time.time()
                    }]
                    
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
    
    st.session_state.conversation_log.append({
        "role": "user", 
        "content": prompt,
        "timestamp": time.time()
    })
    
    # Log user message to both systems
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
            
            # Log to both systems
            if st.session_state.get("airtable_record_id"):
                log_message(
                    st.session_state.airtable_record_id,
                    st.session_state.message_counter,
                    "assistant",
                    closing_response
                )
            
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
            
            # Show session end options
            show_session_end_message()
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
```

if **name** == “**main**”:
main()
