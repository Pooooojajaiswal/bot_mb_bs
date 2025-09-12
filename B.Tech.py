

import os
import streamlit as st
from dotenv import load_dotenv
import logging

from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import AgentRunner
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core import ListIndex
import qdrant_client
from googleapiclient.discovery import build
from typing import List
from pydantic import BaseModel, Field
from llama_index.core.prompts import PromptTemplate



# --- PYDANTIC MODELS FOR QUIZ GENERATION ---

class QuizQuestion(BaseModel):
    """Data model for a single B.Tech-style multiple-choice question."""
    question: str = Field(description="The full text of the quiz question.")
    options: List[str] = Field(description="A list of 4 potential answers (A, B, C, D).")
    correct_answer: str = Field(description="The single correct answer, matching one of the options exactly.")
    explanation: str = Field(description="A brief but detailed explanation for why this is the correct answer.")

class QuizContainer(BaseModel):
    """A container data model that holds a list of quiz questions."""
    questions: List[QuizQuestion]

# --- Setup ---
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
except Exception as e:
    st.error(f"Error loading environment variables: {str(e)}")
    st.stop()

# # --- Llama-Index and Qdrant Configuration ---
# QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# QDRANT_COLLECTION_NAME = "previous_year_questions"

# --- Llama-Index and Qdrant Configuration ---
QDRANT_PATH = "./qdrant_data" # Point to the same local storage path
QDRANT_COLLECTION_NAME = "B.Tech_prep"

# --- DEFINE THE SYSTEM PROMPT ---

SYSTEM_PROMPT = """
## Identity & Role
- You are an expert AI tutor for B.Tech students across **all branches** (Computer Science, Mechanical, Civil, Electrical, Electronics, etc.).
- Tone: Clear, motivating, professional, and approachable.
- Goal: To help students **master engineering concepts and problems** by blending theory, step-by-step problem solving, and real-world applications.
cs

## Core Logic: How to Respond
When a student asks something, decide how to respond:

1. **Is it a Specific Engineering Problem/Code/Calculation/Design Task?**
Examples: "Frame shear force diagram for this beam," "Solve for current in this circuit," "Write code for DFS traversal," "Find COP of a refrigerator cycle."
→ Use **[Problem-Solving Protocol]**


2. **Is it a General Concept/Definition/Principle?**
Examples: "What is entropy?" "Explain Binary Trees," "Define Bernoulli’s theorem."
→ Use **[Foundational Concept Protocol]**

---
### [Protocol 1: Problem-Solving Protocol]
---

**Part 1: Real-World Connection**
- Start with a relatable scenario from engineering practice (bridge load design, embedded system bug, database deadlock, refrigeration cycle, etc.). Explain every aspect of the scenario and why this problem matters in real engineering contexts.


**Part 2: Visual Aid**
- Use the `image_search` tool to fetch ONE **diagram, flowchart, circuit, graph, structural drawing, or schematic**.
- Show it with `![](URL)` and explain why it matters.
- Always explain the visual in detail.


**Part 3: Detailed Step-by-Step Solution**
1. **Restate the Problem Clearly** : Summarize the problem in your own words to ensure clarity.
2. **Relevant Theory & Formulae** : List and explain the key concepts, laws, or formulas needed.
3. **Step-by-Step Derivation/Calculation/Algorithm**: Break down the solution into clear, numbered steps with detailed explanations of each. If you generate code, then explain each section of the code.
4. **Common Mistakes, Pitfalls & Quick Tricks**: Warn about typical errors and share tips to avoid them.
5. **Real-Life Engineering Applications** (industry, research, or practical usage): Briefly mention where this type of problem/solution is applied in real engineering work.


**Part 4: Exam-Oriented Practice**
- Give a university exam/GATE-style problem with a solved step-by-step answer.
- Always ensure the problem matches the B.Tech syllabus, and is common in engineering exams.
- Suggest exam level questions with answers for further practice.


**Part 5: Reflection Check**
- Ask one deep thinking question + one small applied variation (e.g., "What if the beam support changes to a cantilever?" or "What if input is changed to a dynamic array?").


---
### [Protocol 2: Foundational Concept Protocol]
---


**Part 1: Concept Overview**
- Give the textbook-quality definition and explain its importance across engineering disciplines.
- Use a real-world example or analogy to illustrate the concept's relevance (e.g., entropy in thermodynamics, recursion in algorithms, stress-strain in materials).


**Part 2: Visual Blueprint**
- Fetch ONE visual (diagram, architecture, flowchart, graph, schematic) with `image_search`.
- Display it with `![](URL)` and explain every part of the visual in detail.


**Part 3: Step-by-Step Breakdown**
- Explain systematically (theory, principles, formulas, diagrams, or algorithmic flow).
- Explain the theorical part in detail, then move to formulas or algorithmic steps if applicable.
- Use numbered steps or bullet points for clarity.
- Include common misconceptions, pitfalls, and quick tips for mastering the concept.


**Part 4: Link to Real-World Applications**
- Show where the concept is practically applied in engineering systems, research, or industry.
- Give specific examples (e.g., how binary trees are used in databases, how stress-strain concepts apply in civil engineering).
- Relate questions to exam patterns and real engineering problems.


**Part 5: Understanding Check**
- Ask one reflective question and one small applied variation.
- Example: "How does this concept influence system design?" or "What changes if we apply this in a different context?".


---
## Mathematics Formatting Rules
- **NO LATEX EVER** (e.g., \frac, \times, \overline, ^{}, \sqrt{} breaks responses). Use plain text/Unicode readable in any chat window.
- Exponents: Unicode (², ³, ⁴) for numbers; "to the power n" for variables (e.g., x to the power n).
- Inverse trig: Use (sin⁻¹x), never \sin^{-1}x.
- Fractions: numerator / denominator with parentheses: (x² - y²) / (x² + y²).
- Multiplication: Space or · for clarity: 2 x y, 3·sin x.
- Roots: √(...), ∛(...).
- Conjugate: conj(z) or "conjugate of z", never overlines.
- Steps: Bullet-point transformations in a text code block.
- Example (Wrong, FORBIDDEN): f(x) = x^{3}e^{2x}, \frac{1}{x^{2}}, 2(\sin^{-1}x) \cdot \frac{1}{\sqrt{1-x^2}}, 2\times(3/4(x)).
- Example (Correct): For f(x) = x³ e²ˣ, 1/x², 2(sin⁻¹x) · 1/√(1 − x²) − (sin⁻¹x)², 2.(3/4(x)).
- **Self-check**: If LaTeX detected, warn: “LaTeX avoided, using plaintext.” Ensure 100% plaintext/Unicode.



## General Rules & Tool Usage
- **Response**: Never mention Part 1, Part 2, etc. in the final answer. These are for your internal structure only.
- **NO-LATEX**: Never use LaTeX formatting. Always use plain text/Unicode for all math.
- **PYQs:** If the student requests previous year questions (PYQs), use the `previous_year_question_engine` tool.
- **Quiz:** After every major explanation, ask: *“Would you like to test yourself with a quick quiz?”* If yes → use `initiate_adaptive_quiz` tool.
- **Engagement Style:** Always encourage, guide, and keep answers professional yet approachable.
"""

@st.cache_resource
def initialize_system():
    """Initialize all the necessary components for the RAG system."""
    try:
        # --- UPDATE THE LLM INITIALIZATION ---
        Settings.llm = OpenAI(
            model="gpt-4.1-mini", 
            api_key=api_key, 
        )
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=api_key)

        client = qdrant_client.QdrantClient(path=QDRANT_PATH)
        vector_store = QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # --- Tool 1:RAG Query Engine ---
        vector_retriever = index.as_retriever(similarity_top_k=10)
        reranker = SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=3)
        response_synthesizer = get_response_synthesizer(response_mode="compact")
        rag_query_engine = RetrieverQueryEngine(
            retriever=vector_retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[reranker]
        )
        rag_tool = QueryEngineTool(
            query_engine=rag_query_engine,
            metadata=ToolMetadata(
                name="previous_year_question_engine",
                description=(
                    "Use this tool ONLY when the user asks for Previous Year Questions (PYQs). "
                    "This tool searches a database of old exam questions. Do NOT use it for general concept explanations."
                )
            ),
        )
        
        # --- Tool 2: LLM-Only Query Engine ---
        empty_index = ListIndex([])
        llm_query_engine = empty_index.as_query_engine(response_mode="generation")
        llm_tool = QueryEngineTool(
            query_engine=llm_query_engine,
            metadata=ToolMetadata(
                name="textbook_knowledge_engine",
                description=(
                    "The primary tool for explaining B.Tech concepts (theory, derivations, algorithms, problem-solving). "
                    "Use this to define terms, explain mechanisms, formulas, or any general academic question related to the B.Tech syllabus. "
                    "Use this by default unless the user specifically asks for a PYQ or an image."
                )
            ),
        )
        
        # --- IMAGE RETRIEVAL WEB ------
        def search_web_for_image(query: str) -> str:
            """
            Searches the web for an image using the Google Custom Search API,
            and cycles through results to avoid repetition on the same query.
            """
            print(f"\n--- TOOL CALLED [IMAGE SEARCH] ---")
            print(f"Original Query: '{query}'")
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                cse_id = os.getenv("GOOGLE_CSE_ID")
                if not api_key or not cse_id:
                    return "Error: Google API credentials are not configured."
                
                # Get the history and update the count for this query
                query_count = st.session_state.image_search_history.get(query, 0)
                st.session_state.image_search_history[query] = query_count + 1

                # Use the count to fetch a different result from the top 5 images.
                # The start index for Google API is 1-based.
                start_index = 1 + (query_count % 5)
                print(f"Fetching result number {start_index} for this query to ensure variety.")
                # --- END MODIFICATION ---

                service = build("customsearch", "v1", developerKey=api_key)
                res = service.cse().list(
                    q=query,
                    cx=cse_id,
                    searchType='image',
                    num=1,
                    start=start_index,  # Use the calculated start index for variety
                    safe='high'
                ).execute()
                
                if 'items' in res and len(res['items']) > 0:
                    image_url = res['items'][0]['link']
                    return f"IMAGE_URL::{image_url}"  
                else:
                    # Fallback...
                    if start_index > 1:
                        print(f"Fallback: No result at index {start_index}. Trying the top result.")
                        res_fallback = service.cse().list(q=query, cx=cse_id, searchType='image', num=1, start=1, safe='high').execute()
                        if 'items' in res_fallback and len(res_fallback['items']) > 0:
                            return res_fallback['items'][0]['link'] # <-- CHANGED
                    return "Could not find a suitable image for that query."

                # if 'items' in res and len(res['items']) > 0:
                #     image_url = res['items'][0]['link']
                #     return f"IMAGE_URL::{image_url}"
                # else:
                #     # Fallback: If we're past the first result and find nothing, try the first result again.
                #     if start_index > 1:
                #         print(f"Fallback: No result at index {start_index}. Trying the top result.")
                #         res_fallback = service.cse().list(q=query, cx=cse_id, searchType='image', num=1, start=1, safe='high').execute()
                #         if 'items' in res_fallback and len(res_fallback['items']) > 0:
                #             return f"IMAGE_URL::{res_fallback['items'][0]['link']}"
                #     return "Could not find a suitable image for that query."

            except Exception as e:
                logger.error(f"Google Image Search Error: {e}", exc_info=True)
                return f"Error searching for image: {str(e)}"
            
            
        # --- (Inside initialize_system(), right before the IMAGE RETRIEVAL section) ---
        
        # --- ADAPTIVE QUIZ FUNCTION STUBS ---
        # These are placeholders. You must implement their real logic.
        
        def get_weakest_topics(profile: dict, subject: str, count: int) -> list:
            """
            STUB FUNCTION: Replace with logic to get weak topics.
            Placeholder topics are now B.Tech-relevant.
            """
            if subject == "Computer Science":
                return ["Data Structures", "Operating Systems"]
            elif subject == "Electronics":
                return ["Digital Logic", "Signals & Systems"]
            elif subject == "Mechanical":
                return ["Thermodynamics", "Strength of Materials"]
            return ["Engineering Mathematics", "Basic Electrical"]


        def generate_interactive_test_with_llm(llm, subject, question_count, focus_topics, difficulty_mix) -> list:
            """
            REAL FUNCTION (B.Tech MODIFIED): Calls the LLM to generate a structured list of quiz questions
            by forcing the output into our Pydantic schema. Removed 'student_class'.
            """
            print(f"[REAL FUNCTION - B.Tech] Generating {question_count} questions for topics: {focus_topics}")

            # Combine the topics into a clean string for the prompt
            topics_string = ", ".join(focus_topics)
            
            # --- THIS IS THE MODIFIED B.Tech PROMPT TEMPLATE ---
            quiz_prompt_template_str = """
            You are an expert B.Tech AI tutor and question setter.
            Generate {count} high-quality, university/GATE-style multiple-choice questions.

            RULES:
            - Subject: {subject}
            - Main Topics: {topics}
            - All questions must match the B.Tech curriculum.
            - Options must be plausible, with one correct answer.
            - Explanations must be concise and technical.
            - Correct answer text must exactly match one of the options.
            """
            
            prompt_template = PromptTemplate(quiz_prompt_template_str)

            try:
                
                response = llm.structured_predict(
                    QuizContainer,  # The Pydantic class we want it to fill
                    prompt_template,  # The NEW B.Tech prompt template
                    subject=subject,
                    topics=topics_string,
                    count=question_count
                )
          
                
                question_list_of_dicts = [q.model_dump() for q in response.questions]
                
                if not question_list_of_dicts:
                    raise ValueError("LLM returned an empty question list.")

                print(f"Successfully generated {len(question_list_of_dicts)} real B.Tech questions.")
                return question_list_of_dicts

            except Exception as e:
                logger.error(f"Error in REAL quiz generation: {e}", exc_info=True)
                # Fallback to a placeholder if the structured generation fails
                return [
                    {
                        "question": "Sorry, I had an error generating a real question. This is a fallback.",
                        "options": ["Option A", "Option B (Error)", "Option C", "Option D"],
                        "correct_answer": "Option B (Error)",
                        "explanation": f"The generation failed with: {e}"
                    }
                ]
            
        # --- YOUR NEW TOOL FUNCTION DEFINITION (B.Tech MODIFIED) ---
        def initiate_adaptive_quiz(subject: str, topic: str) -> str:
            """
            Initiates a short, adaptive quiz for the user on a specific B.Tech topic.
            This tool should be called by the agent after getting the user's consent.
            Requires the current 'subject' (inferred) and the specific 'topic' of the lesson.
            """
            print(f"\n--- TOOL CALLED [B.Tech ADAPTIVE QUIZ GEN] for Subject: {subject}, Topic: {topic} ---")
            
            # 1. The conversational topic is the highest priority "seed" for the quiz.
            seed_topic = topic

            # 2. Check the profile for other weak topics to potentially mix in (using new B.Tech stubs).
            weak_topics_from_profile = get_weakest_topics(
                profile=st.session_state.performance_profile, 
                subject=subject, 
                count=2
            )

            # 3. Build the final list of topics.
            focus_topics = [seed_topic]
            for t in weak_topics_from_profile:
                if t.lower() != seed_topic.lower() and len(focus_topics) < 3:
                    focus_topics.append(t)
    
            print(f"B.Tech Quiz generator focusing on topics: {', '.join(focus_topics)}")

            # 4. Directly call the (B.Tech-modified) question generator
            # NOTE: We are generating 3 questions now and 'student_class' is REMOVED.
            quiz_questions = generate_interactive_test_with_llm(
                llm=Settings.llm,  # Using the LLM from our Settings
                subject=subject,
                question_count=3,   # Let's make the B.Tech quiz slightly longer
                focus_topics=focus_topics,
                difficulty_mix={'Medium': 2, 'Hard': 1} # B.Tech questions are generally harder
            )

            if not quiz_questions:
                return "I'm sorry, I had a little trouble creating a quiz on that specific topic right now. Let's continue our lesson."

            # 5. Set up the quiz state.
            st.session_state.current_test = quiz_questions
            st.session_state.current_question_index = 0
            st.session_state.conversation_mode = "quiz" # This is the critical state change
            st.session_state.quiz_responses = [] # To track answers
    
            # This response goes back to the LLM, not the user. The UI will change based on the state.
            return f"A {len(quiz_questions)}-question quiz on {', '.join(focus_topics)} has been generated. The UI will now display the first question."

            
            # This creates the tool object that the agent can use.
        image_retrieval_tool = FunctionTool.from_defaults(
        fn=search_web_for_image,
            name="image_search",
            description=(
                "Searches Google Images for a diagram, histology slide, radiology scan, or biochemical pathway relevant to an B.Tech syllabus topic. "
                "Receives a specific search query (e.g., 'histology of caseous necrosis') and returns a single direct image URL. "

            )
        )
        
        # --- Tool 3: Adaptive Quiz Tool ---
        quiz_tool = FunctionTool.from_defaults(
            fn=initiate_adaptive_quiz,
            name="initiate_adaptive_quiz",
            description=(
                "Use this tool to create and start an adaptive quiz for the user. "
                "This should only be called AFTER the user agrees to take a quiz. "
                "You must pass the current 'subject' (e.g., 'Biology') and the specific 'topic' just discussed (e.g., 'Enzyme Action')."
            )
        )

        all_tools = [rag_tool, image_retrieval_tool, quiz_tool]
        agent = AgentRunner.from_llm(
            tools=all_tools,
            llm=Settings.llm,  
            system_prompt=SYSTEM_PROMPT, 
            verbose=True  # Setting verbose=True is great for debugging in your terminal
        )
        
        return agent 

    except Exception as e:
        st.error(f"Failed to initialize the RAG system: {str(e)}")
        logger.error(f"Initialization error: {str(e)}", exc_info=True)
        st.stop()

# --- Streamlit UI ---
st.title("B.Tech Prep Coach")
st.write("Ask me to explain any concept, I can be your learning buddy for your B.Tech journey!")

# Initialize system
query_engine = initialize_system()

# --- CLEAN UP SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Namaste! I'm your dedicated B.Tech coach. What you want to look at today?"
        }
    ]
if "image_search_history" not in st.session_state:
    st.session_state.image_search_history = {}
if "subject_selected" not in st.session_state:
    st.session_state.subject_selected = None
if "conversation_mode" not in st.session_state:
    st.session_state.conversation_mode = "chat"  # Default mode is chat
if "current_test" not in st.session_state:
    st.session_state.current_test = []
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "quiz_responses" not in st.session_state:
    st.session_state.quiz_responses = []
if "performance_profile" not in st.session_state:
    st.session_state.performance_profile = {} # Your quiz logic will populate this
if "show_quiz_feedback" not in st.session_state:
    st.session_state.show_quiz_feedback = None

# Display conversation history
# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- MODE-BASED UI LOGIC ---
# ===================================================================
# STATE 1: QUIZ MODE (REVISED LOGIC)
# ===================================================================
if st.session_state.conversation_mode == "quiz":
    
    # Check if the quiz is valid and we haven't finished
    if st.session_state.current_test and st.session_state.current_question_index < len(st.session_state.current_test):
        
        q = st.session_state.current_test[st.session_state.current_question_index]
        
        # This is the key: check if we are showing feedback OR asking the question
        if st.session_state.show_quiz_feedback:
            # --- STATE B: SHOWING FEEDBACK ---
            # We have already submitted an answer, so just show the results.
            
            feedback_data = st.session_state.show_quiz_feedback
            
            st.markdown(f"### Result for Question {st.session_state.current_question_index + 1}")
            st.markdown(f"**{q['question']}**")
            
            # Show the user's selected answer (you can format this however you like)
            st.write(f"*Your answer: {feedback_data['selected']}*")

            # Display the correct/incorrect message AND the explanation
            if feedback_data['is_correct']:
                st.success(f"Correct! \n\n*Explanation: {q['explanation']}*")
            else:
                st.error(f"Not quite. The correct answer is **{q['correct_answer']}**. \n\n*Explanation: {q['explanation']}*")
            
            # Now show the button to advance
            if st.button("Next Question"):
                st.session_state.current_question_index += 1 # Advance index
                st.session_state.show_quiz_feedback = None # Reset feedback state
                st.rerun() # Rerun to show next question or summary

        else:
            # --- STATE A: ASKING THE QUESTION ---
            # This is the default state: show the question and options.
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
                    # User submitted. Store the response...
                    st.session_state.quiz_responses.append({
                        "question": q['question'],
                        "selected": user_answer,
                        "correct": q['correct_answer'],
                        "explanation": q['explanation']
                    })
                    
                    # (This is where you would update the performance_profile)

                    # ...NOW, INSTEAD OF ADVANCING, just store feedback data...
                    st.session_state.show_quiz_feedback = {
                        "selected": user_answer,
                        "is_correct": user_answer == q['correct_answer']
                    }
                    
                    # ...and rerun to trigger STATE B (showing feedback).
                    st.rerun()
                else:
                    st.warning("Please select an answer.")
    
    else:
        # Quiz is over (This logic was correct)
        st.success("Quiz complete!")
        # ... (rest of your quiz summary logic is fine)
        # ...
        st.session_state.conversation_mode = "chat"
        st.session_state.current_test = []
        st.session_state.quiz_responses = []
        st.session_state.show_quiz_feedback = None # Clean up feedback state

        st.session_state.messages.append({"role": "assistant", "content": "Great job on the quiz! What concept would you like to cover next?"})
        if st.button("Back to Lesson"): # Use if to avoid error on auto-rerun
            st.rerun()


# ===================================================================
# STATE 2: CHAT MODE 
# This only runs if we are NOT in quiz mode
# ===================================================================
elif st.session_state.conversation_mode == "chat":
        
    user_input = st.chat_input(f"Ask your query:")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        contextual_query = f"[Subject Context: {st.session_state.subject_selected}] User query: {user_input}"

                
        logger.info(f"Sending contextual query to agent: {contextual_query}")

        # Generate and display response from the agent
        try:
            with st.spinner(f"Thinking..."):
                response = query_engine.chat(contextual_query)
                response_content = str(response)

            # Add assistant's response to history
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            # Display assistant's response
            with st.chat_message("assistant"):
                st.markdown(response_content)
                
                # Rerun to clear the input box and show the full response
            st.rerun() 
                    
        except Exception as e:
            error_message = f"Sorry, an error occurred: {str(e)}"
            st.error(error_message)
            logger.error(f"Query processing error: {str(e)}", exc_info=True)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

