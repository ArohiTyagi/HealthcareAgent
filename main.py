import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

# ============================================
# LOAD ENV VARIABLES
# ============================================

load_dotenv()

# ============================================
# OLLAMA API CONFIGURATION (CREWAI)
# ============================================

# Using CrewAI's native LLM integration for Ollama
llm = LLM(
    model="ollama/llama3", 
    base_url="http://localhost:11434"
)

# ============================================
# CREWAI AGENTS & TASKS
# ============================================

analyzer = Agent(
    role='Medical Symptom Analyst',
    goal='Analyze symptoms and explain them clearly without creating panic.',
    backstory='You are an experienced healthcare assistant. You understand common symptoms and explain them clearly.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

advisor = Agent(
    role='Health & Wellness Advisor',
    goal='Provide beginner-friendly health advice, hydration tips, rest recommendations, food suggestions, and precautions.',
    backstory='You are a friendly health advisor who focuses on practical, easy-to-follow lifestyle advice.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

safety_officer = Agent(
    role='Patient Safety Officer',
    goal='Identify emergency symptoms and recommend doctor consultation when needed.',
    backstory='You are responsible for patient safety. You calmly identify emergency symptoms without creating panic.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

formatter = Agent(
    role='Medical Communications Specialist',
    goal='Organize healthcare responses into a clean, structured and readable format.',
    backstory='You are a Response Formatter. You organize healthcare responses into a clean, structured and readable format.',
    llm=llm,
    allow_delegation=False,
    verbose=True
)

analyzer_task = Task(
    description='Analyze the following symptoms carefully: {symptoms}\n Mention: possible common causes, likely conditions, and a simple explanation. Keep response short and calm.',
    expected_output='A short and calm analysis of the symptoms, listing common causes and likely conditions.',
    agent=analyzer
)

advisor_task = Task(
    description='Give healthcare advice for the symptoms: {symptoms}\n Include: hydration advice, food suggestions, rest recommendations, and precautions. Keep response simple and beginner-friendly.',
    expected_output='Practical healthcare advice including hydration, food, rest, and precautions.',
    agent=advisor
)

safety_task = Task(
    description='Check whether these symptoms need urgent medical attention: {symptoms}\n Mention: warning signs, emergency symptoms, and if doctor consultation is needed. Avoid creating panic.',
    expected_output='A safety assessment indicating if there are warning signs or if a doctor consultation is needed.',
    agent=safety_officer
)

formatter_task = Task(
    description='Create a final healthcare response using all previous analyses.\n Structure:\n 1. Symptom Analysis\n 2. Healthcare Advice\n 3. Safety Check\n Keep it clean and readable.',
    expected_output='A cleanly formatted healthcare response with 3 numbered sections.',
    agent=formatter
)

healthcare_crew = Crew(
    agents=[analyzer, advisor, safety_officer, formatter],
    tasks=[analyzer_task, advisor_task, safety_task, formatter_task],
    process=Process.sequential,
    verbose=True
)

# ============================================
# STREAMLIT UI
# ============================================

st.set_page_config(page_title="Healthcare Chatbot", page_icon="🏥", layout="centered")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1 {
        color: #4CAF50;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 LangChain Healthcare Chatbot")
st.markdown("<p style='text-align: center; color: #888;'>Powered by LangChain & Ollama</p>", unsafe_allow_html=True)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Enter Symptoms or Health Question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process with CrewAI
    with st.chat_message("assistant"):
        with st.spinner("Analyzing symptoms and gathering advice using CrewAI (this may take a moment)..."):
            
            try:
                final_result = healthcare_crew.kickoff(inputs={"symptoms": user_input})
                final_response = str(final_result) + "\n\n---\n**⚠ DISCLAIMER:** This chatbot is for educational purposes only. Consult a licensed medical professional for accurate diagnosis or treatment."
                
                st.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})
            except Exception as e:
                st.error(f"Error communicating with Ollama or CrewAI: {e}")