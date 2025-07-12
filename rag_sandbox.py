import streamlit as st
import pandas as pd
from typing import List, Dict

def embed_test_question(question: str, get_embedding_func) -> List[float]:
    """
    Embed a test question for RAG evaluation
    """
    try:
        return get_embedding_func(question)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return []

def query_pinecone_for_sandbox(index, question: str, get_embedding_func, top_k: int = 5) -> List[Dict]:
    """
    Query Pinecone for sandbox testing with detailed metadata
    """
    try:
        question_vector = embed_test_question(question, get_embedding_func)
        if not question_vector:
            return []
        
        results = index.query(
            vector=question_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for i, match in enumerate(results.matches):
            chunk_data = {
                'rank': i + 1,
                'score': round(match.score, 4),
                'text': match.metadata.get('text_preview', 'No text available')[:500],
                'full_text': match.metadata.get('text_preview', 'No text available'),
                'source': match.metadata.get('source_url', 'Unknown'),
                'topics': match.metadata.get('tennis_topics', 'General'),
                'skill_level': match.metadata.get('skill_level', 'Not specified'),
                'coaching_style': match.metadata.get('coaching_style', 'Not specified'),
                'vector_id': match.id
            }
            chunks.append(chunk_data)
        
        return chunks
        
    except Exception as e:
        st.error(f"Pinecone query error: {e}")
        return []

def get_claude_response_with_context(question: str, chunks: List[Dict], claude_client) -> str:
    """
    Get Claude response using retrieved chunks as context
    """
    try:
        # Build context from chunks
        context_sections = []
        for chunk in chunks:
            context_sections.append(f"""
Resource {chunk['rank']}:
Topics: {chunk['topics']}
Level: {chunk['skill_level']}
Style: {chunk['coaching_style']}
Content: {chunk['text']}
""")
        
        context_text = "\n".join(context_sections)
        
        prompt = f"""You are a professional tennis coach. Use the provided coaching resources to answer the question accurately and helpfully.

Professional Coaching Resources:
{context_text}

Question: "{question}"

Provide a comprehensive tennis coaching response using the above resources:"""

        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Error generating response with context: {e}"

def get_claude_response_without_context(question: str, claude_client) -> str:
    """
    Get Claude response using only general knowledge (no RAG context)
    """
    try:
        prompt = f"""You are a professional tennis coach. Answer this tennis question using your general knowledge only.

Question: "{question}"

Provide a tennis coaching response:"""

        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Error generating response without context: {e}"

def display_chunks_analysis(chunks: List[Dict]):
    """
    Display retrieved chunks with detailed analysis
    """
    if not chunks:
        st.warning("No chunks retrieved for this question.")
        return
    
    st.markdown("### 📚 Retrieved Chunks Analysis")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score = sum(chunk['score'] for chunk in chunks) / len(chunks)
        st.metric("Average Relevance", f"{avg_score:.3f}")
    with col2:
        high_relevance = len([c for c in chunks if c['score'] > 0.8])
        st.metric("High Relevance (>0.8)", high_relevance)
    with col3:
        unique_sources = len(set(chunk['source'] for chunk in chunks))
        st.metric("Unique Sources", unique_sources)
    
    # Detailed chunk breakdown
    for chunk in chunks:
        relevance_color = "🟢" if chunk['score'] > 0.8 else "🟡" if chunk['score'] > 0.6 else "🔴"
        
        with st.expander(f"{relevance_color} Rank #{chunk['rank']} | Score: {chunk['score']} | {chunk['topics']}"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Content Preview:**")
                st.write(chunk['text'] + ("..." if len(chunk['full_text']) > 500 else ""))
                
                if st.button(f"Show Full Text", key=f"full_text_{chunk['rank']}"):
                    st.markdown("**Full Content:**")
                    st.write(chunk['full_text'])
            
            with col2:
                st.markdown("**Metadata:**")
                st.write(f"**Source:** {chunk['source']}")
                st.write(f"**Topics:** {chunk['topics']}")
                st.write(f"**Level:** {chunk['skill_level']}")
                st.write(f"**Style:** {chunk['coaching_style']}")
                st.write(f"**Vector ID:** `{chunk['vector_id'][:8]}...`")

def display_response_comparison(question: str, with_context: str, without_context: str):
    """
    Display side-by-side comparison of Claude responses
    """
    st.markdown("### 🔍 Response Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟩 Claude **WITH** Pinecone Context")
        st.markdown("*Using retrieved coaching resources*")
        st.write(with_context)
        
        # Response analysis
        word_count_with = len(with_context.split())
        st.caption(f"Word count: {word_count_with}")
        
    with col2:
        st.markdown("#### 🟦 Claude **WITHOUT** Context")
        st.markdown("*Using general knowledge only*")
        st.write(without_context)
        
        # Response analysis
        word_count_without = len(without_context.split())
        st.caption(f"Word count: {word_count_without}")
    
    # Quick comparison metrics
    st.markdown("#### 📊 Quick Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        length_diff = word_count_with - word_count_without
        st.metric("Length Difference", f"{length_diff:+d} words")
    
    with col2:
        # Check for specific tennis terms
        tennis_terms = ['forehand', 'backhand', 'serve', 'volley', 'grip', 'stance', 'footwork', 'topspin', 'slice']
        with_terms = sum(1 for term in tennis_terms if term.lower() in with_context.lower())
        without_terms = sum(1 for term in tennis_terms if term.lower() in without_context.lower())
        st.metric("Tennis Terms (With/Without)", f"{with_terms}/{without_terms}")
    
    with col3:
        # Simple specificity check
        specific_words = ['specific', 'exactly', 'precisely', 'particular', 'detailed']
        with_specific = sum(1 for word in specific_words if word.lower() in with_context.lower())
        without_specific = sum(1 for word in specific_words if word.lower() in without_context.lower())
        st.metric("Specificity Words", f"{with_specific}/{without_specific}")

def display_rag_sandbox_interface(index, claude_client, get_embedding_func):
    """
    Main RAG Sandbox interface
    """
    st.markdown("# 🧪 RAG Performance Sandbox")
    st.markdown("Test and evaluate your Pinecone knowledge retrieval system")
    st.markdown("---")
    
    # Sandbox controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        test_question = st.text_input(
            "🎾 Enter a tennis coaching question:",
            placeholder="How do I improve my forehand consistency?",
            help="Ask any tennis-related question to test RAG performance"
        )
    
    with col2:
        top_k = st.selectbox("Chunks to retrieve:", [3, 5, 8, 10], index=1)
    
    if st.button("🔍 Test RAG Performance", type="primary", disabled=not test_question):
        if test_question:
            with st.spinner("🧠 Processing question through RAG pipeline..."):
                
                # Step 1: Embed and query Pinecone
                st.markdown("### 🎯 Step 1: Question Embedding & Retrieval")
                chunks = query_pinecone_for_sandbox(index, test_question, get_embedding_func, top_k)
                
                if chunks:
                    st.success(f"✅ Retrieved {len(chunks)} chunks from Pinecone")
                    
                    # Step 2: Display chunks analysis
                    display_chunks_analysis(chunks)
                    
                    # Step 3: Get both responses
                    st.markdown("### 🤖 Step 2: Generating Claude Responses")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.spinner("Getting response WITH context..."):
                            response_with_context = get_claude_response_with_context(test_question, chunks, claude_client)
                    
                    with col2:
                        with st.spinner("Getting response WITHOUT context..."):
                            response_without_context = get_claude_response_without_context(test_question, claude_client)
                    
                    # Step 4: Display comparison
                    display_response_comparison(test_question, response_with_context, response_without_context)
                    
                else:
                    st.error("❌ No chunks retrieved. Check your question or Pinecone index.")
    
    # Example questions for testing
    st.markdown("---")
    st.markdown("### 💡 Example Test Questions")
    
    example_questions = [
        "How do I improve my forehand consistency?",
        "What's the best grip for a beginner's serve?",
        "How can I develop better footwork?",
        "What drills help with volley technique?",
        "How do I overcome tennis anxiety?",
        "What's the difference between topspin and slice?",
        "How often should beginners take lessons?",
        "What equipment do I need to start playing tennis?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {question}", key=f"example_{i}"):
                st.session_state.sandbox_question = question
                st.rerun()
    
    # Auto-fill if example was clicked
    if st.session_state.get('sandbox_question'):
        st.info(f"Selected: {st.session_state.sandbox_question}")
        if st.button("Clear Selection"):
            del st.session_state.sandbox_question
            st.rerun()
