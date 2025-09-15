import os
import streamlit as st
from dotenv import load_dotenv
import logging
from typing import List
from pydantic import BaseModel, Field

# --- LlamaIndex / Qdrant Imports ---
from llama_index.core import VectorStoreIndex, get_response_synthesizer, ListIndex, Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters 
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import AgentRunner
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
import qdrant_client

from llama_index.core import SimpleDirectoryReader
from llama_index.core.vector_stores import FilterCondition
import tempfile

# --- Google Search Tool Imports ---
from googleapiclient.discovery import build

# Import the catalog and prompt logic from our new files
from _config import COURSE_CATALOG
from _prompts import get_system_prompt

import logging
logger = logging.getLogger(__name__)

# --- PYDANTIC MODELS FOR QUIZ GENERATION (Unchanged) ---

class QuizQuestion(BaseModel):
    """Data model for a single multiple-choice question."""
    topic: str = Field(description="The specific topic this question relates to, must be one of the topics provided.")
    question: str = Field(description="The full text of the quiz question.")
    options: List[str] = Field(description="A list of 4 potential answers (A, B, C, D).")
    correct_answer: str = Field(description="The single correct answer, matching one of the options exactly.")
    explanation: str = Field(description="A brief but detailed explanation for why this is the correct answer.")

class QuizContainer(BaseModel):
    """A container data model that holds a list of quiz questions."""
    questions: List[QuizQuestion]
    
    
IMAGE_SEARCH_EXCLUSION_LIST = [
    "pinterest.com",
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "chegg.com",
    "coursehero.com",
    "scribd.com"
]

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    if not google_api_key or not google_cse_id:
        st.warning("Google API Key or CSE ID not found. Image search tool will be disabled.")
except Exception as e:
    st.error(f"Error loading environment variables: {str(e)}")
    st.stop()
    
    
Settings.llm = OpenAI(model="gpt-4.1-mini", api_key=api_key)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)

# --- Qdrant Configuration ---
QDRANT_PATH = "./qdrant_college_data"  # Local persistent storage path
DOC_CHAT_COLLECTION = "user_document_collection"

DOCUMENT_CHAT_PROMPT_TEMPLATE = PromptTemplate(
"""
You are an expert AI assistant who is a master at analyzing and summarizing documents in an understandable format.  
Your goal is to provide clear, detailed, and helpful answers to the query based ONLY on the context provided.  

## Your Task:
A user has asked a question about a document they uploaded. You have been given the most relevant 
snippets of text from that document as context. You must use this context to answer the question.

## Response Instructions:
- Begin with a concise, direct answer to the user’s question.  
- Incorporate quotes or phrases from the provided context naturally within the explanation (do not label them as "supporting evidence").  
- Expand on the answer with additional clarification and explanation as needed, making the response easy to follow and well-structured.  
- When listing multiple items (e.g., different types, steps, or categories), present them as **bullet points** for readability.  
- Always format the response in clean **Markdown** with proper spacing, line breaks, and list formatting.  
- Do not use headings like "Direct Answer," "Supporting Evidence," or "Detailed Explanation" in the final output.  

## Critical Rules:
- If the query is not clear (e.g., contains vague terms like 'I', 'hmm') or ambiguous, ask for clarification.  
- If the context contains code, format it using Markdown with appropriate syntax highlighting.  
- You are forbidden from using any knowledge outside of the provided context.  
- If the context does not contain the information needed to answer the question, you MUST state: "Based on the provided text, I could not find an answer to that question." and do not provide any further explanation.  

---------------------  
CONTEXT FROM DOCUMENT:  
{context_str}  
---------------------  
USER'S QUESTION: {query_str}  
YOUR STRUCTURED RESPONSE:  
"""
)

@st.cache_resource
def get_qdrant_client():
    """
    Initializes and returns a single, cached instance of the Qdrant client.
    """
    logger.info("Initializing Qdrant client...")
    return qdrant_client.QdrantClient(path=QDRANT_PATH)

# ##########################################################################
# --- GLOBAL TOOL FUNCTIONS (Unchanged) ---
# ##########################################################################

def search_web_for_image(query: str) -> str:
    """
    Searches the web for an image using the Google Custom Search API.
    Cycles through results to avoid repetition on the same query.
    """
    logger.info(f"--- TOOL CALLED [IMAGE SEARCH] Query: '{query}' ---")
    try:
        if not google_api_key or not google_cse_id:
            logger.warning("Google credentials not found. Image search skipped.")
            return "Error: Image search credentials are not configured."
        
        # Create the exclusion string (e.g., "-site:pinterest.com -site:facebook.com")
        exclusion_string = " ".join([f"-site:{site}" for site in IMAGE_SEARCH_EXCLUSION_LIST])
        
        # Combine the agent's query with our hard-coded exclusion filters
        modified_query = f"{query} {exclusion_string}"
        logger.info(f"--- Modified Query with Exclusions: '{modified_query}' ---")
        
        query_count = st.session_state.image_search_history.get(query, 0)
        st.session_state.image_search_history[query] = query_count + 1
        start_index = 1 + (query_count % 5) # Get top 5 results in a cycle

        service = build("customsearch", "v1", developerKey=google_api_key)
        res = service.cse().list(
            q=modified_query,
            cx=google_cse_id,
            searchType='image',
            num=1,
            start=start_index,
            safe='high'
        ).execute()
        
        if 'items' in res and len(res['items']) > 0:
            image_url = res['items'][0]['link']
            return f"IMAGE_URL::{image_url}"
        else:
            if start_index > 1: # Fallback to first result
                res_fallback = service.cse().list(
                    q=modified_query, 
                    cx=google_cse_id, 
                    searchType='image', 
                    num=1, 
                    start=1, 
                    safe='high'
                ).execute()
                
                if 'items' in res_fallback and len(res_fallback['items']) > 0:
                    # The fallback must also return the same format
                    fallback_url = res_fallback['items'][0]['link']
                    return f"IMAGE_URL::{fallback_url}"
                
            return "Could not find a suitable image for that query."

    except Exception as e:
        logger.error(f"Google Image Search Error: {e}", exc_info=True)
        return f"Error searching for image: {str(e)}"
    
def find_document_relationships(doc_filenames: List[str]) -> str:
    """
    Analyzes two or more documents to find relationships, overlaps, or contradictions.
    This tool should be used when the user asks a general question about how documents are related.
    """
    logger.info(f"--- TOOL CALLED [RELATIONSHIP FINDER] for: {doc_filenames} ---")
    if len(doc_filenames) < 2:
        return "Please select at least two documents to compare their relationships."

    # Use the existing index and a summary engine to get the gist of each document
    client = get_qdrant_client()
    vector_store = QdrantVectorStore(client=client, collection_name=DOC_CHAT_COLLECTION)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    summaries = {}
    for filename in doc_filenames:
        logger.info(f"Generating summary for {filename}...")
        # Create a filter for each document
        doc_filter = MetadataFilters(filters=[ExactMatchFilter(key="file_name", value=filename)])
        
        # Create a temporary query engine to summarize just this one document
        summary_engine = index.as_query_engine(
            response_mode="tree_summarize",
            filters=doc_filter
        )
        summary_response = summary_engine.query(f"Provide a detailed summary of the key topics, arguments, and conclusions in the document '{filename}'.")
        summaries[filename] = str(summary_response)

    # Now, use the LLM to compare the summaries
    all_summaries_text = "\n\n".join([f"## Summary of '{fname}':\n{text}" for fname, text in summaries.items()])
    
    comparison_prompt = PromptTemplate(
    """
    You are a research analyst. Your task is to compare the following document summaries.
    Based ONLY on the text provided, describe the relationship between these documents.
    Consider the following:
    - Do they cover the same topic from different perspectives?
    - Is one document an extension or a critique of the other?
    - Are there any clear points of overlap or contradiction?
    - If there is no clear thematic or topical relationship, you MUST state that they appear to be unrelated.

    Here are the summaries:
    ---------------------
    {summaries}
    ---------------------

    Your final analysis:
    """
    )
    
    # Use the global LLM from Settings
    response = Settings.llm.predict(comparison_prompt, summaries=all_summaries_text)
    return response

def update_performance_profile(profile: dict, subject: str, topic: str, is_correct: bool):
    """
    Safely finds or creates the entry for a given subject/topic and 
    updates its performance stats.
    """
    try:
        # 1. Ensure 'subjects' key exists
        if "subjects" not in profile:
            profile["subjects"] = {}
            
        # 2. Find or create the subject entry
        if subject not in profile["subjects"]:
            profile["subjects"][subject] = {"topics": {}, "subject_summary": {}}
            
        # 3. Find or create the topic entry within that subject
        if topic not in profile["subjects"][subject]["topics"]:
            profile["subjects"][subject]["topics"][topic] = {
                "performance": {"correct": 0, "incorrect": 0, "total": 0, "accuracy_percentage": 0.0},
                "study_sessions": [],
                "total_study_time_minutes": 0
            }
        
        # 4. Get the paths to the data we need to update
        perf_stats = profile["subjects"][subject]["topics"][topic]["performance"]
        
        # 5. Increment the stats
        perf_stats["total"] += 1
        if is_correct:
            perf_stats["correct"] += 1
        else:
            perf_stats["incorrect"] += 1
            
        # 6. Recalculate accuracy
        perf_stats["accuracy_percentage"] = round((perf_stats["correct"] / perf_stats["total"]) * 100, 2)
        
        logger.info(f"Profile updated for [{subject}][{topic}]: {perf_stats}")
        
    except Exception as e:
        logger.error(f"Failed to update performance profile: {e}", exc_info=True)

    # Note: This function modifies the 'profile' dict in-place.

# We should still keep a minimum threshold to avoid ranking topics based on just one or two questions.
MIN_ATTEMPTS_FOR_RANKING = 3

def get_weakest_topics(profile: dict, subject: str, count: int) -> list:
    """
    Analyzes the user's complex performance profile (JSON structure) to find their 
    weakest topics for a given subject.
    
    It navigates to profile['subjects'][subject]['topics'] and checks the 
    'performance' block for each topic.
    """
    logger.info(f"[REAL LOGIC] Analyzing complex profile for weakest topics in '{subject}'...")
    
    # 1. Navigate down to the 'subjects' dictionary within the profile
    subjects_dict = profile.get("subjects", {})
    
    # 2. Get the specific data block for the requested subject
    subject_data = subjects_dict.get(subject)
    
    if not subject_data:
        # This handles cases where the user has no history at all for this subject
        logger.warning(f"No performance data found for subject '{subject}'. Returning empty list.")
        return []
        
    # 3. Get the dictionary of all topics within that subject
    topics_dict = subject_data.get("topics", {})
    if not topics_dict:
        logger.warning(f"No topics found within subject '{subject}'. Returning empty list.")
        return []

    # 4. Iterate over all topics and collect their accuracy scores if they meet the threshold
    topic_accuracies = []
    for topic_name, topic_details in topics_dict.items():
        
        # 5. Get the nested 'performance' block
        performance_stats = topic_details.get("performance", {})
        if not performance_stats:
            # Skip this topic if it has no performance data block
            continue 

        total_attempts = performance_stats.get("total", 0)
        
        # 6. Only rank topics where the user has a significant attempt history
        if total_attempts >= MIN_ATTEMPTS_FOR_RANKING:
            
            # 7. Read the pre-calculated accuracy directly from the JSON
            accuracy = performance_stats.get("accuracy_percentage", 0.0) 
            topic_accuracies.append((topic_name, accuracy))
    
    if not topic_accuracies:
        logger.warning(f"No topics in '{subject}' meet the min attempt threshold ({MIN_ATTEMPTS_FOR_RANKING}).")
        return []

    # 8. Sort the list by accuracy (ASCENDING) to put the weakest topics (lowest %) first
    topic_accuracies.sort(key=lambda item: item[1])  # item[1] is the accuracy_percentage

    # 9. Extract just the names from the sorted list
    weakest_topic_names = [topic[0] for topic in topic_accuracies]
    
    # 10. Return the top 'count' topics from this weak-list
    final_list = weakest_topic_names[:count]
    logger.info(f"Found weakest topics: {final_list}")
    return final_list

def generate_interactive_test_with_llm(llm, subject: str, focus_topics: List[str]) -> List[dict]:
    """
    REAL FUNCTION: Calls the LLM to generate a structured quiz.
    This prompt is now generic and takes the subject/topics as variables.
    """
    logger.info(f"[REAL FUNCTION] Generating quiz for topics: {focus_topics}")
    topics_string = ", ".join(focus_topics)
    
    quiz_prompt_template_str = """
    You are an expert AI Coach and university-level exam question creator.
    Your task is to generate {count} high-quality, multiple-choice questions.

    RULES:
    - Subject: {subject}
    - Main Topics: {topics}
    - All questions MUST be accurate and relevant to a college-level curriculum for this subject.
    - You MUST provide the specific 'topic' for EACH question, matching one of the requested topics.
    - Options must be plausible, and explanations clear, concise, and academic.
    - Ensure 'correct_answer' text EXACTLY matches one string in the 'options' list.
    - Difficulty: Medium.

    Generate the quiz now.
    """
    prompt_template = PromptTemplate(quiz_prompt_template_str)

    try:
        response = llm.structured_predict(
            QuizContainer,
            prompt_template,
            subject=subject,
            topics=topics_string,
            count=3  # Generate a 3-question quiz
        )
        question_list_of_dicts = [q.model_dump() for q in response.questions]
        
        if not question_list_of_dicts:
            raise ValueError("LLM returned an empty question list.")

        logger.info(f"Successfully generated {len(question_list_of_dicts)} real questions.")
        return question_list_of_dicts

    except Exception as e:
        logger.error(f"Error in REAL quiz generation: {e}", exc_info=True)
        return [ # Fallback error question
            {
                "question": "Sorry, I had an error generating a real question.",
                "options": ["Option A", "Option B (Error)", "Option C", "Option D"],
                "correct_answer": "Option B (Error)",
                "explanation": f"The generation failed with: {e}"
            }
        ]

def initiate_adaptive_quiz(subject: str, topic: str) -> str:
    """
    Initiates a short, adaptive quiz. This is the function the LLM calls.
    It now uses the globally-set Settings.llm
    """
    logger.info(f"--- TOOL CALLED [ADAPTIVE QUIZ] for Subject: {subject}, Topic: {topic} ---")
    
    weak_topics_from_profile = get_weakest_topics(
        profile=st.session_state.performance_profile, 
        subject=subject, 
        count=2
    )
    
    focus_topics = [topic]
    for t in weak_topics_from_profile:
        if t.lower() != topic.lower() and len(focus_topics) < 3:
            focus_topics.append(t)

    quiz_questions = generate_interactive_test_with_llm(
        llm=Settings.llm,  # Uses the globally configured LLM
        subject=subject,
        focus_topics=focus_topics
    )

    if not quiz_questions:
        return "Sorry, I had trouble creating a quiz. Let's continue the lesson."

    # Set up the quiz state (your logic is perfect)
    st.session_state.current_test = quiz_questions
    st.session_state.current_question_index = 0
    st.session_state.conversation_mode = "quiz" 
    st.session_state.quiz_responses = [] 
    
    return f"A {len(quiz_questions)}-question quiz on {', '.join(focus_topics)} is ready. The UI will now show the first question."


# ##########################################################################
# --- NEW: HYBRID AGENT BUILDER (Unchanged, but now cleaner) ---
# This function is the same, but it now calls the IMPORTED get_system_prompt
# ##########################################################################

@st.cache_resource(show_spinner=False)
def get_specialized_agent(course: str, field: str, subject: str):
    """
    This function builds and caches a UNIQUE agent for a specific subject.
    It CHECKS if a Qdrant collection exists for it.
    - If YES, it builds a FULL RAG AGENT.
    - If NO, it builds a "LITE" PERSONA-ONLY AGENT.
    """ 
    try:
        # 1. GENERATE COLLECTION NAME & LOAD PROMPT
        collection_name = f"{course}_{field}".lower().replace(" & ", "_").replace(" ", "_")
        
        # MODIFICATION: Calls the imported function now!
        system_prompt = get_system_prompt(course, field) 

        # 2. CONFIGURE LLM SETTINGS
        # llm = OpenAI(model="gpt-4.1-mini", api_key=api_key)
        # Settings.llm = llm # Set globally for tools to use
        
        #    This object will store the conversation history.
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        # 3. CHECK FOR KNOWLEDGE BASE (THE HYBRID LOGIC)
        # client = qdrant_client.QdrantClient(path=QDRANT_PATH)
        client = get_qdrant_client()
        all_collection_names = [c.name for c in client.get_collections().collections]
        
        all_tools = []
        knowledge_tool_created = False

        if collection_name in all_collection_names:
            logger.info(f"SUCCESS: Found Knowledge Base for '{subject}'. Building FULL RAG AGENT.")
            # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)
            
            vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            
            vector_retriever = index.as_retriever(similarity_top_k=10)
            reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)
            response_synthesizer = get_response_synthesizer(response_mode="compact")
            
            rag_query_engine = RetrieverQueryEngine(
                retriever=vector_retriever,
                response_synthesizer=response_synthesizer,
                node_postprocessors=[reranker]
            )
            
            knowledge_base_tool = QueryEngineTool(
                query_engine=rag_query_engine,
                metadata=ToolMetadata(
                    name="knowledge_base_tool",
                    description=(
                        f"Use this tool ONLY to answer factual questions or get specific details from the course textbook for {subject}. "
                        "Prioritize information from this tool above your general knowledge."
                    )
                ),
            )
            all_tools.append(knowledge_base_tool)
            knowledge_tool_created = True

        else:
            logger.warning(f"NOTICE: No Knowledge Base found for '{collection_name}'. Building 'LITE' AGENT.")
        

        # 4. ADD GENERIC TOOLS (Added to ALL agents)
        if google_api_key and google_cse_id:
            image_retrieval_tool = FunctionTool.from_defaults(
                fn=search_web_for_image,
                name="image_search",
                description=(
                    "This is the ONLY tool for finding visual aids like diagrams, charts, or flowcharts. "
                    "This tool is MANDATORY for fulfilling 'Part 2' of both the '[Case/Provision Protocol]' "
                    "and the '[Foundational Concept Protocol]'. Use this by providing a specific search query "
                    "for a relevant legal or procedural diagram."
                    
                )
            )
            all_tools.append(image_retrieval_tool)
        
        quiz_tool = FunctionTool.from_defaults(
            fn=initiate_adaptive_quiz,
            name="initiate_adaptive_quiz",
            description=(
                "Use this tool to create and start a quiz. Call this ONLY AFTER the user agrees to take a quiz. "
                "You must pass the current 'subject' (e.g., 'Biology') and the specific 'topic' just discussed."
            )
        )
        all_tools.append(quiz_tool)
        
        relationship_tool = FunctionTool.from_defaults(
            fn=find_document_relationships,
            name="document_relationship_analyzer",
            description="Use this tool to analyze and describe the relationship between two or more documents."
        )
        all_tools.append(relationship_tool)

        # 5. BUILD AND RETURN THE FINAL AGENT
        agent = AgentRunner.from_llm(
            tools=all_tools,
            llm=Settings.llm,  
            system_prompt=system_prompt, 
            memory=memory,
            verbose=True
        )
        
        return agent 

    except Exception as e:
        st.error(f"Failed to initialize agent for {subject}: {str(e)}")
        logger.error(f"Agent initialization error: {str(e)}", exc_info=True)
        return None

# ##########################################################################
# --- STREAMLIT UI (This logic is unchanged, but now uses imported CATALOG) ---
# ##########################################################################

st.title("University AI Tutor Platform")

# --- Initialize Session State (Unchanged) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "image_search_history" not in st.session_state:
    st.session_state.image_search_history = {}
if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = "chat"
if "current_test" not in st.session_state:
    st.session_state.current_test = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "quiz_responses" not in st.session_state:
    st.session_state.quiz_responses = []
if "performance_profile" not in st.session_state:
    st.session_state.performance_profile = {} 
if "show_quiz_feedback" not in st.session_state:
    st.session_state.show_quiz_feedback = None
if "selected_course" not in st.session_state:
    st.session_state.selected_course = None
if "selected_field" not in st.session_state:
    st.session_state.selected_field = None
if "selected_subject" not in st.session_state:
    st.session_state.selected_subject = None
if "current_agent" not in st.session_state:
    st.session_state.current_agent = None
if "chat_mode_selected" not in st.session_state:
    st.session_state.chat_mode_selected = None # <--  state to track AI vs Document chat
if "document_agent" not in st.session_state:
    st.session_state.document_agent = None # <-- state for the document agent
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# --- Simple User Authentication (for multi-tenancy) ---
if 'user_id' not in st.session_state or not st.session_state.user_id:
    st.info("Welcome! Please enter a username to begin.")
    st.session_state.user_id = st.text_input("Enter your username:", key="user_login")
    if st.session_state.user_id:
        st.rerun()
else:
    # Once logged in, show the main tabs
    tab1, tab2 = st.tabs(["🎓 AI Tutor", "📄 Document Chat"])

    # ===================================================================
    # --- TAB 1: AI TUTOR WORKFLOW ---
    # ===================================================================
    with tab1:
        if not st.session_state.selected_subject:
            
            st.info("Welcome! Please select your course, field, and subject to begin a session.")
            
            courses = list(COURSE_CATALOG.keys())
            sel_course = st.selectbox("Select your Course:", ["---"] + courses, key="sel_course")
            
            fields = []
            if sel_course and sel_course != "---":
                fields = list(COURSE_CATALOG[sel_course].keys())
            sel_field = st.selectbox("Select your Field:", ["---"] + fields, key="sel_field", disabled=(len(fields) == 0))

            subjects = []
            if sel_course != "---" and sel_field and sel_field != "---":
                subjects = COURSE_CATALOG[sel_course][sel_field]
            sel_subject = st.selectbox("Select your Subject:", ["---"] + subjects, key="sel_subject", disabled=(len(subjects) == 0))

            if st.button("Start AI Tutor Session", key="start_tutor"):
                if sel_course != "---" and sel_field != "---" and sel_subject != "---":
                    st.session_state.selected_course = sel_course
                    st.session_state.selected_field = sel_field
                    st.session_state.selected_subject = sel_subject
                    st.session_state.conversation_mode = "chat"
                    st.session_state.messages = [] # Clear messages for the new session

                    with st.spinner("Preparing your AI Tutor..."):
                        st.session_state.current_agent = get_specialized_agent(
                            course=st.session_state.selected_course,
                            field=st.session_state.selected_field,
                            subject=st.session_state.selected_subject
                        )
                    
                    # MODIFICATION: Only create a welcome message if the agent loaded successfully
                    if st.session_state.current_agent:
                        welcome_msg = f"AI Tutor session started for **{st.session_state.selected_subject}**. Ask me anything!"
                        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
                    
                    st.rerun()        
                    
    # --- STATE 2: AI TUTOR CHAT INTERFACE ---
        else:
            st.caption(f"Current Session: {st.session_state.selected_course} > {st.session_state.selected_field} > {st.session_state.selected_subject}")
            if st.button("End Tutor Session", key="end_tutor"):
                # More targeted reset for the tutor session
                st.session_state.selected_subject = None
                st.session_state.selected_course = None
                st.session_state.selected_field = None
                st.session_state.current_agent = None
                st.session_state.messages = []
                st.session_state.conversation_mode = "chat"
                st.rerun()
            
            # --- MODIFICATION: Add an explicit check to see if the agent is loaded ---
            if not st.session_state.current_agent:
                st.error(
                    "The AI Tutor agent failed to initialize. "
                    "This can happen due to an invalid API key, a model name issue, or a problem with your prompts. "
                    "Please check your terminal for the full error message and then click 'End Tutor Session' to try again."
                )
            else:
                # This is the main UI that runs ONLY if the agent is ready
                
                # Display chat history
                for message in st.session_state.get("messages", []):
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Handle the different conversation modes (chat vs quiz)
                if st.session_state.get("conversation_mode") == "chat":
                    user_input = st.chat_input(f"Ask your {st.session_state.selected_subject} question:")
                    if user_input:
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        
                        try:
                            with st.spinner(f"Thinking about {st.session_state.selected_subject}..."):
                                response = st.session_state.current_agent.chat(user_input)
                                response_content = response.response 
                            
                            # MODIFICATION: Add robustness check for empty responses
                            if response_content and response_content.strip():
                                st.session_state.messages.append({"role": "assistant", "content": response_content})
                            else:
                                logger.warning("AI Tutor Agent returned an empty response.")
                                error_message = "I'm sorry, I couldn't generate a response. Please try rephrasing."
                                st.session_state.messages.append({"role": "assistant", "content": error_message})
                        
                        except Exception as e:
                            error_message = f"Sorry, a critical error occurred. Please check the terminal for details."
                            logger.error(f"Query processing error: {str(e)}", exc_info=True)
                            st.session_state.messages.append({"role": "assistant", "content": error_message})
                        
                        st.rerun()
                    
                elif st.session_state.get("conversation_mode") == "quiz":
                    
                    if st.session_state.current_test and st.session_state.current_question_index < len(st.session_state.current_test):
                            
                            q = st.session_state.current_test[st.session_state.current_question_index]
                            
                            if st.session_state.show_quiz_feedback:
                                feedback_data = st.session_state.show_quiz_feedback
                                st.markdown(f"### Result for Question {st.session_state.current_question_index + 1}")
                                st.markdown(f"**{q['question']}**")
                                st.write(f"*Your answer: {feedback_data['selected']}*")

                                if feedback_data['is_correct']:
                                    st.success(f"Correct! \n\n*Explanation: {q['explanation']}*")
                                else:
                                    st.error(f"Not quite. The correct answer is **{q['correct_answer']}**. \n\n*Explanation: {q['explanation']}*")
                                
                                if st.button("Next Question"):
                                    st.session_state.current_question_index += 1 
                                    st.session_state.show_quiz_feedback = None 
                                    st.rerun()

                            else:
                                st.markdown(f"### Question {st.session_state.current_question_index + 1} of {len(st.session_state.current_test)}")
                                st.markdown(f"**{q['question']}**")
                                
                                user_answer = st.radio(
                                    "Select your answer:",
                                    q['options'],
                                    index=None,
                                    key=f"quiz_q_{st.session_state.current_question_index}"
                                )
                                
                                if st.button("Submit Answer"):
                                    if user_answer:

                                        # Determine if the answer was correct (we just make it a variable)
                                        is_correct_answer = (user_answer == q['correct_answer'])
                                        
                                        # Get the topic from the question data (This relies on you adding 'topic' 
                                        #    to your Pydantic model and prompt, as per my Step 1)
                                        question_topic = q.get("topic") # 'q' is the dict for the current question
                                        
                                        # Get the currently selected subject from session state
                                        current_subject = st.session_state.selected_subject

                                        #  It saves this one result back into your main st.session_state.performance_profile
                                        if question_topic and current_subject:
                                            # This calls the new helper function you added to the global scope
                                            update_performance_profile(
                                                profile=st.session_state.performance_profile, # The main profile
                                                subject=current_subject,
                                                topic=question_topic,
                                                is_correct=is_correct_answer
                                            )
                                        else:
                                            # This log helps you debug if your Pydantic/prompt modification in Step 1 didn't work
                                            logger.warning("Could not update profile: Missing topic in quiz question data or subject in session state.")
                                        
                                        # This adds the response to a temporary list (good for an end-of-quiz summary)
                                        st.session_state.quiz_responses.append({
                                            "question": q['question'], 
                                            "selected": user_answer,
                                            "correct": q['correct_answer'], 
                                            "explanation": q['explanation'],
                                            "topic": question_topic # <-- It's a good idea to store the topic here too
                                        })
                                        st.session_state.show_quiz_feedback = {
                                            "selected": user_answer,
                                            "is_correct": user_answer == q['correct_answer'] 
                                        }
                                        st.rerun()
                                    else:
                                        st.warning("Please select an answer.")
                        
                    else:
                            st.success("Quiz complete!")
                            st.session_state.conversation_mode = "chat"
                            st.session_state.current_test = []
                            st.session_state.quiz_responses = []
                            st.session_state.show_quiz_feedback = None 

                            st.session_state.messages.append({"role": "assistant", "content": "Great job on the quiz! What concept would you like to cover next?"})
                            if st.button("Back to Lesson"): 
                                st.rerun()

                            pass
                    
    with tab2:
        st.header("Chat With Your Documents")
        st.info("Upload new documents or select from your saved library to begin.")
        with st.sidebar:
                st.header("Your Documents")

                # --- 1. LIST EXISTING DOCUMENTS FROM QDRANT ---
                try:
                    client = get_qdrant_client() # Use our cached client
                    # Check if the main document collection exists
                    collections = client.get_collections().collections
                    collection_names = [c.name for c in collections]
                    
                    user_doc_filenames = []
                    if DOC_CHAT_COLLECTION in collection_names:
                        # Scroll through points to find all unique file_name values in the metadata
                        response = client.scroll(
                            collection_name=DOC_CHAT_COLLECTION,
                            limit=1000, # Adjust as needed
                            with_payload=["file_name"],
                            with_vectors=False
                        )
                        # Use a set to get unique filenames
                        seen_files = set()
                        for record in response[0]:
                            if record.payload and "file_name" in record.payload:
                                seen_files.add(record.payload["file_name"])
                        user_doc_filenames = sorted(list(seen_files))

                except Exception as e:
                    st.error("Could not connect to document database.")
                    logger.error(f"Qdrant connection error: {e}")
                    user_doc_filenames = []

                selected_doc_filenames = st.multiselect("Select document(s) to chat with:", user_doc_filenames)

                # --- 2. LOAD THE AGENT WITH A METADATA FILTER ---
                if selected_doc_filenames: # Check if the list is not empty
                    # Create a user-friendly string of selected filenames for the button
                    filenames_str = ", ".join([f"'{f}'" for f in selected_doc_filenames])
                    
                    if st.button(f"Load {filenames_str}"):
                        with st.spinner(f"Loading and preparing agent for {filenames_str}..."):
                            client = get_qdrant_client()
                            vector_store = QdrantVectorStore(client=client, collection_name=DOC_CHAT_COLLECTION)
                            
                            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
                            
                            # --- NEW: Build a filter that accepts multiple documents ---
                            

                            # Create an ExactMatchFilter for each selected filename
                            individual_filters = [
                                ExactMatchFilter(key="file_name", value=fname) 
                                for fname in selected_doc_filenames
                            ]
                            
                            # Combine them with an OR condition, so it retrieves chunks from ANY of the selected files
                            filters = MetadataFilters(
                                filters=individual_filters, 
                                condition=FilterCondition.OR
                            )
                                        
                            vector_query_engine = index.as_query_engine(filters=filters)
                            
                            vector_tool = QueryEngineTool(
                                query_engine=vector_query_engine,
                                metadata=ToolMetadata(
                                    name="vector_search_tool",
                                    description=f"Use this tool to answer specific questions about the content of the document '{selected_doc_filenames}'."
                                ),
                            )

                            # Tool 2: The Summary Tool (for high-level questions)
                            # This tool can look at the whole document to summarize
                            summary_query_engine = index.as_query_engine(
                                response_mode="tree_summarize",
                                filters=filters # Still filter by filename
                            )
                            summary_tool = QueryEngineTool(
                                query_engine=summary_query_engine,
                                metadata=ToolMetadata(
                                    name="summary_tool",
                                    description=f"Use this tool to summarize the document, list topics, or answer other high-level questions about '{selected_doc_filenames}'."
                                ),
                            )
                            
                            # --- NEW: Create a true AgentRunner with the tools ---
                            
                            # We don't need a complex persona, just instructions on how to use the tools
                            doc_agent_prompt = f"""
                            You are an expert AI assistant who is a master at analyzing the document titled '{selected_doc_filenames}'.  
                            Your primary goal is to provide detailed, comprehensive, and well-structured answers based ONLY on the information returned by your tools.  

                            ## Your Persona:
                            - Act as a helpful research assistant.  
                            - Be thorough and methodical in your explanations.  
                            - Never use outside knowledge. Your expertise is limited to the provided document.  

                            ## How to Handle Queries:
                            1. **Tool Selection**: You have a 'summary_tool' for high-level questions (like "list all topics", "summarize the document") and a 'vector_search_tool' for specific, detailed questions ("what is a clustered index?"). Choose the best tool for the user's query.  
                            2. **Typo Correction**: If you suspect a typo in the user's query (e.g., "Coddy Rule" instead of "Codd's Rule"), proactively use the corrected term with your tools. In your answer, clearly state that you have made the correction. For example: *"I didn’t find 'Coddy Rule,' but here is the information for the likely intended topic, 'Codd’s Rule': ..."*  
                            3. **Handle Greetings**: If the user provides a simple greeting like 'hi' or 'hello', respond with a polite, brief greeting and state your purpose (e.g., 'Hello! I'm ready to answer questions about your documents.'). Do not use your tools for greetings.

                            ## Response Instructions:
                            - Always start with the **definition or description** of the concept directly from the document (quote or paraphrase it naturally).  
                            - Then provide a **clear, well-structured explanation** of the concept.  
                            - When relevant, expand with **examples, applications, or comparisons** to make the idea more understandable.  
                            - For multiple elements (like types, steps, or rules), present them as **bullet points or numbered lists** and explain each point briefly.  
                            - Keep the formatting clean and in **Markdown** for readability.  
                            - Do not use section headings like “Summary Answer,” “Key Points,” or “Conclusion.” The flow should feel natural, like a well-written article or teaching note.  

                            ## Critical Rules:
                            - If the query is unclear or ambiguous, politely ask for clarification.  
                            - If the context contains code, format it using Markdown with proper syntax highlighting.  
                            - You are forbidden from using any knowledge outside of the provided document.  
                            - If the context does not contain the needed information, you must state:  
                            *"Based on the provided text, I could not find an answer to that question."* and nothing more.  

    

                            """

                            doc_agent = AgentRunner.from_llm(
                                tools=[vector_tool, summary_tool],
                                llm=Settings.llm,
                                system_prompt=doc_agent_prompt, # Use the new, detailed prompt
                                verbose=True
                            )
                            
                            st.session_state.document_agent = doc_agent
                            # -------------------------------------------------------------------
                            
                            st.session_state.messages = [{"role": "assistant", "content": f"Loaded agent for '{selected_doc_filenames}'. You can now ask me specific or high-level questions about it."}]
                            st.rerun()


                st.divider()

                # --- 3. UPLOAD NEW DOCUMENTS (WITH METADATA) ---
                st.header("Upload a New Document")
                uploaded_file = st.file_uploader(
                    "Upload a PDF or TXT to save it to the collection", type=['pdf', 'txt']
                )

                if uploaded_file:
                    if st.button("Process and Save Document"):
                        with st.spinner(f"Processing and saving '{uploaded_file.name}'..."):
                            with tempfile.TemporaryDirectory() as temp_dir:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                
                                # Load the document
                                documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
                                # *** CRITICAL: Add the filename as metadata to each chunk ***
                                for doc in documents:
                                    doc.metadata["file_name"] = uploaded_file.name
                                
                                client = get_qdrant_client()
                                vector_store = QdrantVectorStore(client=client, collection_name=DOC_CHAT_COLLECTION)
                                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                                
                                # Create or update the index with the new, metadata-tagged documents
                                VectorStoreIndex.from_documents(
                                    documents, storage_context=storage_context
                                )
                        st.success(f"Successfully saved '{uploaded_file.name}' to the collection!")
                        st.rerun() # Rerun to update the dropdown list
                        pass

        if st.session_state.get("document_agent"):
            # Display chat history for Document Chat
            # Use a separate key for messages to keep conversations isolated
            if "doc_messages" not in st.session_state:
                st.session_state.doc_messages = [{"role": "assistant", "content": "Agent is loaded. Ask a question about the selected document(s)."}]
                
            for message in st.session_state.doc_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                
            # Handle chat input
            doc_user_input = st.chat_input("Ask a question about the loaded document...")
            if doc_user_input:
                st.session_state.doc_messages.append({"role": "user", "content": doc_user_input})
                    
                with st.spinner("Analyzing document..."):
                    response = st.session_state.document_agent.chat(doc_user_input)
                    response_content = response.response
                    
                st.session_state.doc_messages.append({"role": "assistant", "content": response_content})
                st.rerun()
        else:
            st.info("Please use the sidebar to upload a new document or load a saved one.")

