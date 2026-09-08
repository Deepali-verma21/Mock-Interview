import streamlit as st
import pandas as pd
import json
import os

from components.interview_engine import InterviewSession, PERSONA_PROMPTS
from components.resume_parser import extract_text_from_pdf, extract_skills_and_summary
from components.question_bank import get_filtered_questions, QUESTION_BANK
from components.code_sandbox import execute_python_code, analyze_code_with_ai, CODING_PRESETS, run_preset_test_suite
from components.evaluator import generate_performance_eval, create_radar_chart
from components.voice_utils import render_speech_to_text_widget, render_browser_tts, speak_text, transcribe_audio_bytes, TRANSCRIPTION_UNAVAILABLE

# Page Configuration
st.set_page_config(
    page_title="AI Mock Interview Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern dark UI
st.markdown("""
<style>
    .stApp {
        background-color: #0f111a;
        color: #e0e6ed;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #89b4fa !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #a6adc8 !important;
    }

    .chat-interviewer {
        background-color: #1e2238;
        border-left: 4px solid #89b4fa;
        padding: 14px 18px;
        border-radius: 0 12px 12px 0;
        margin-bottom: 8px;
        color: #cdd6f4;
        font-size: 1.02rem;
    }
    
    .chat-candidate {
        background-color: #1b2b34;
        border-right: 4px solid #a6e3a1;
        padding: 14px 18px;
        border-radius: 12px 0 0 12px;
        margin-bottom: 8px;
        color: #e6edf3;
        font-size: 1.02rem;
        text-align: right;
    }

    .badge-status {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-gemini {
        background-color: #1e3a8a;
        color: #93c5fd;
        border: 1px solid #3b82f6;
    }
    .badge-sim {
        background-color: #374151;
        color: #d1d5db;
        border: 1px solid #6b7280;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #161927;
        border-radius: 8px;
        color: #a6adc8;
        padding: 0px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2b304a !important;
        color: #89b4fa !important;
        border-bottom: 2px solid #89b4fa !important;
    }
    
    .test-pass {
        background-color: #1c3326;
        color: #a6e3a1;
        padding: 8px 14px;
        border-radius: 6px;
        border: 1px solid #2e5b3e;
        margin-bottom: 6px;
        font-size: 0.9em;
    }
    .test-fail {
        background-color: #3b1e2b;
        color: #f38ba8;
        padding: 8px 14px;
        border-radius: 6px;
        border: 1px solid #682638;
        margin-bottom: 6px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "api_key" not in st.session_state:
    st.session_state.api_key = os.environ.get("GEMINI_API_KEY", "")

if "gemini_client" not in st.session_state:
    st.session_state.gemini_client = None
    if st.session_state.api_key:
        try:
            from google import genai
            st.session_state.gemini_client = genai.Client(api_key=st.session_state.api_key)
        except Exception:
            st.session_state.gemini_client = None

if "interview_session" not in st.session_state or not hasattr(st.session_state.interview_session, "get_elapsed_minutes"):
    st.session_state.interview_session = None

if "eval_report" not in st.session_state:
    st.session_state.eval_report = None

if "resume_parsed_data" not in st.session_state:
    st.session_state.resume_parsed_data = None

if "qbank_ratings" not in st.session_state:
    st.session_state.qbank_ratings = {}

if "transcribed_text_buffer" not in st.session_state:
    st.session_state.transcribed_text_buffer = ""

if "mic_reset_counter" not in st.session_state:
    st.session_state.mic_reset_counter = 0

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/brain.png", width=64)
    st.title("Interview AI Control")
    
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Enter API key for live Gemini 2.5 response generation."
    )
    
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        if api_key_input:
            try:
                from google import genai
                st.session_state.gemini_client = genai.Client(api_key=api_key_input)
                st.success("API Key updated!")
            except Exception as e:
                st.error(f"Invalid Key: {e}")
                st.session_state.gemini_client = None
        else:
            st.session_state.gemini_client = None
            
    if st.session_state.gemini_client:
        st.markdown('<span class="badge-status badge-gemini">✨ Gemini 2.5 Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-status badge-sim">⚡ Simulation Mode Active</span>', unsafe_allow_html=True)

    st.caption("Build: debug-diag-v1")
        
    st.markdown("---")
    st.subheader("Interview Configuration")
    
    selected_role = st.selectbox(
        "Select Interview Domain",
        options=[
            "Software Engineering",
            "System Design Architect",
            "Data Science & ML",
            "Product Manager",
            "Behavioral (STAR Method)"
        ]
    )
    
    selected_difficulty = st.select_slider(
        "Seniority Level",
        options=["Junior", "Mid-Level", "Senior", "Lead / Staff"]
    )
    
    st.markdown("---")
    
    if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
        st.session_state.interview_session = InterviewSession(
            role=selected_role,
            difficulty=selected_difficulty,
            client=st.session_state.gemini_client
        )
        st.session_state.interview_session.start_interview()
        st.session_state.eval_report = None
        st.session_state.transcribed_text_buffer = ""
        st.session_state.mic_reset_counter += 1
        st.rerun()

# Main Application Layout
st.title("🎯 AI-Based Mock Interview Platform")
st.caption("Practice real-time technical, system design, and behavioral interviews with persona-driven AI evaluation.")

tab_interview, tab_resume, tab_sandbox, tab_qbank, tab_analytics = st.tabs([
    "🎯 Live Mock Interview",
    "📄 Resume Scanner & Tailor",
    "💻 Coding Sandbox",
    "📚 Question Bank",
    "📊 Performance Analytics"
])

# TAB 1: Live Mock Interview
with tab_interview:
    if st.session_state.interview_session is None or not hasattr(st.session_state.interview_session, "get_elapsed_minutes"):
        st.session_state.interview_session = InterviewSession(
            role=selected_role,
            difficulty=selected_difficulty,
            client=st.session_state.gemini_client
        )
        st.session_state.interview_session.start_interview()
        
    session = st.session_state.interview_session

    # Keep the active session's client in sync with the sidebar. Without
    # this, entering/changing the API key mid-session has no effect until
    # "Start New Interview" is clicked, because the session object was
    # constructed once with whatever client existed at creation time.
    session.client = st.session_state.gemini_client
    
    # Live Header Metrics
    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    with col_p1:
        st.markdown(f"**Domain:** `{session.role}` | **Level:** `{session.difficulty}`")
    with col_p2:
        progress_val = min(1.0, session.question_count / session.max_questions)
        st.progress(progress_val, text=f"Question {session.question_count} of {session.max_questions}")
    with col_p3:
        elapsed = session.get_elapsed_minutes() if hasattr(session, "get_elapsed_minutes") else 0.0
        st.markdown(f"⏱️ **Elapsed Time:** `{elapsed} min`")

    # Surface the real reason live grading isn't running, instead of the
    # generic "simulation mode" message hiding what actually failed.
    # Always-visible debug panel — not conditional on an error existing, so
    # you can confirm what's actually running in the deployed app rather
    # than guessing whether the fix took effect.
    with st.expander("🔧 Debug: AI Grading Status", expanded=True):
        st.write("Sidebar client is set:", st.session_state.gemini_client is not None)
        st.write("Sidebar client type:", type(st.session_state.gemini_client).__name__)
        st.write("Active session's client is set:", getattr(session, "client", None) is not None)
        st.write("Active session's client type:", type(getattr(session, "client", None)).__name__)
        st.write("session.last_error attribute exists:", hasattr(session, "last_error"))
        st.write("session.last_error value:", getattr(session, "last_error", "N/A - attribute missing, old code is running"))

    if getattr(session, "last_error", None):
        st.error(f"Live grading call failed, so this fell back to simulation mode. Error: `{session.last_error}`")

    st.markdown("---")
    
    # Render Interactive Chat History
    for idx, msg in enumerate(session.chat_history):
        if msg["role"] == "interviewer":
            st.markdown(f'<div class="chat-interviewer">{msg["text"]}</div>', unsafe_allow_html=True)
            render_browser_tts(msg["text"], button_label="🔊 Read Question Out Loud")
        else:
            st.markdown(f'<div class="chat-candidate"><b>You</b>: {msg["text"]}</div>', unsafe_allow_html=True)
                
    st.markdown("---")
    
    # Interactive Input & Shortcuts
    if not session.is_completed:
        st.subheader("Your Response")
        
        # Native Streamlit Audio Microphone Input with Delete / Reset Button
        col_mic1, col_mic2 = st.columns([3, 1])
        with col_mic1:
            recorded_audio = st.audio_input("🎙️ Record Voice Answer (Click mic icon to record)", key=f"native_mic_{st.session_state.mic_reset_counter}")
        with col_mic2:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🗑️ Delete Audio & Re-record", use_container_width=True):
                st.session_state.mic_reset_counter += 1
                st.session_state.transcribed_text_buffer = ""
                st.toast("Audio recording & transcript cleared. You can record again!", icon="🧹")
                st.rerun()

        if recorded_audio is not None:
            with st.spinner("Transcribing your spoken answer..."):
                audio_bytes = recorded_audio.read()
                transcription = transcribe_audio_bytes(audio_bytes, client=st.session_state.gemini_client)
                if transcription == TRANSCRIPTION_UNAVAILABLE:
                    st.warning(
                        "Couldn't transcribe that recording (no active Gemini key and offline speech "
                        "recognition failed). Your answer box was NOT auto-filled — please type your "
                        "answer or use the Web Speech Dictation box below."
                    )
                elif transcription and transcription != st.session_state.transcribed_text_buffer:
                    st.session_state.transcribed_text_buffer = transcription
                    st.session_state.mic_reset_counter += 1
                    st.success("Audio transcribed successfully! You can edit or re-record if needed.")
                    st.rerun()

        with st.expander("🌐 Web Speech API Dictation (Browser Dictation Alternative)", expanded=False):
            render_speech_to_text_widget()
            
        final_answer_to_submit = ""
        default_val = st.session_state.transcribed_text_buffer
        
        if session.role == "Behavioral (STAR Method)":
            st.markdown("#### 🌟 Guided STAR Response Builder")
            with st.expander("Click to open STAR framework helper fields", expanded=True):
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    star_s = st.text_area("1. Situation (Context & background)", height=70, placeholder="What was the challenge or setting?")
                    star_t = st.text_area("2. Task (Your specific responsibility)", height=70, placeholder="What goal were you assigned to achieve?")
                with col_s2:
                    star_a = st.text_area("3. Action (Specific steps YOU took)", height=70, placeholder="What decision, PoC, or leadership action did you perform?")
                    star_r = st.text_area("4. Result (Quantifiable outcome & learnings)", height=70, placeholder="What was the measurable metric or result?")
                    
                assembled_star = f"**Situation:** {star_s}\n**Task:** {star_t}\n**Action:** {star_a}\n**Result:** {star_r}"
                
            candidate_input = st.text_area(
                "Or type/paste full response directly:",
                value=default_val,
                height=100,
                placeholder="Type your response or recorded text will populate here...",
                key=f"candidate_answer_input_star_{st.session_state.mic_reset_counter}"
            )
            final_answer_to_submit = candidate_input.strip() if candidate_input.strip() else assembled_star.strip()
        else:
            candidate_input = st.text_area(
                "Type or edit your response:",
                value=default_val,
                height=120,
                placeholder="Structure your solution, approach, Big-O complexity, and key architectural choices...",
                key=f"candidate_answer_input_normal_{st.session_state.mic_reset_counter}"
            )
            final_answer_to_submit = candidate_input.strip()

        st.markdown("**⚡ Interactive Action Shortcuts:**")
        col_act1, col_act2, col_act3, col_act4 = st.columns(4)
        
        action_override = None
        with col_act1:
            if st.button("🚀 Submit Answer", type="primary", use_container_width=True):
                action_override = None
                if final_answer_to_submit:
                    with st.spinner("Interviewer is evaluating response..."):
                        session.process_candidate_answer(final_answer_to_submit)
                        st.session_state.transcribed_text_buffer = ""
                        st.session_state.mic_reset_counter += 1
                        st.rerun()
                else:
                    st.warning("Please type or speak your answer before submitting.")

        with col_act2:
            if st.button("⚡ Challenge Complexity", use_container_width=True):
                # Ensure the candidate provides a substantive answer before challenging complexity
                if final_answer_to_submit and len(final_answer_to_submit.strip()) > 10:
                    with st.spinner("Interviewer is challenging complexity..."):
                        session.process_candidate_answer(final_answer_to_submit, action_override="Challenge My Complexity")
                        st.session_state.transcribed_text_buffer = ""
                        st.session_state.mic_reset_counter += 1
                        st.rerun()
                else:
                    st.warning("Please provide a detailed explanation of how you would optimize the solution before challenging the complexity.")

        with col_act3:
            if st.button("⚠️ Ask Edge Case", use_container_width=True):
                with st.spinner("Interviewer is introducing edge cases..."):
                    session.process_candidate_answer(final_answer_to_submit if final_answer_to_submit else "How does this handle edge cases?", action_override="Ask Edge Case Scenario")
                    st.session_state.transcribed_text_buffer = ""
                    st.session_state.mic_reset_counter += 1
                    st.rerun()

        with col_act4:
            if st.button("💡 Request Hint", use_container_width=True):
                last_q = [m["text"] for m in session.chat_history if m["role"] == "interviewer"][-1]
                st.info(session.generate_hint(last_q))
    else:
        st.success("🎉 Interview completed! Click below to view your full scorecard & detailed evaluation.")
        if st.button("📊 Generate & View Scorecard", type="primary", use_container_width=True):
            with st.spinner("Analyzing transcript & compiling performance report..."):
                report = generate_performance_eval(
                    session.chat_history,
                    session.role,
                    session.difficulty,
                    client=st.session_state.gemini_client
                )
                st.session_state.eval_report = report
                st.info("Scorecard ready! Switch to the 'Performance Analytics' tab to view your full report.")

# TAB 2: Resume Scanner & Tailor
with tab_resume:
    st.header("📄 Resume Analysis & Custom Question Generator")
    st.write("Upload your resume (PDF or Text) to parse your top technical skills, experience level, and generate tailored interview practice questions.")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            bytes_data = uploaded_file.read()
            raw_text = extract_text_from_pdf(bytes_data)
        else:
            raw_text = uploaded_file.read().decode("utf-8")
            
        with st.expander("🔍 Extracted Raw Resume Text", expanded=False):
            st.text_area("Raw Text", raw_text, height=180)
            
        if st.button("⚡ Parse Resume & Tailor Questions", type="primary"):
            with st.spinner("Extracting candidate profile and key engineering skills..."):
                parsed = extract_skills_and_summary(raw_text, client=st.session_state.gemini_client)
                st.session_state.resume_parsed_data = parsed
                st.success("Resume parsed successfully!")
                
    if st.session_state.resume_parsed_data:
        data = st.session_state.resume_parsed_data
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.subheader("🎯 Extracted Profile Summary")
            st.markdown(f"**Assessed Level:** `{data.get('experience_level', 'Mid-Level')}`")
            st.markdown("**Identified Key Skills:**")
            skills_html = " ".join([f"<span style='background:#2e344e; color:#89b4fa; padding:4px 10px; border-radius:12px; margin-right:6px;'>{s}</span>" for s in data.get("skills", [])])
            st.markdown(skills_html, unsafe_allow_html=True)
            
            st.markdown("**Key Achievements / Experience:**")
            for proj in data.get("key_projects", []):
                st.markdown(f"- {proj}")
                
        with col_r2:
            st.subheader("📌 Tailored Practice Questions")
            tailored_qs = data.get("tailored_questions", [])
            for idx, q in enumerate(tailored_qs, 1):
                st.markdown(f"**Q{idx}:** {q}")
                if st.button(f"Start Interview with Q{idx}", key=f"start_q_{idx}"):
                    st.session_state.interview_session = InterviewSession(
                        role=selected_role,
                        difficulty=selected_difficulty,
                        client=st.session_state.gemini_client
                    )
                    st.session_state.interview_session.start_interview(custom_starter=q)
                    st.success("Started tailored interview! Switch to the 'Live Mock Interview' tab.")

# TAB 3: Coding Sandbox & Automated Test Runner
with tab_sandbox:
    st.header("💻 Interactive Coding Sandbox & Unit Test Runner")
    st.write("Select a coding problem preset or write custom Python algorithms. Run automated test suites and get instant AI complexity feedback.")
    
    selected_preset = st.selectbox("Load Problem Preset", list(CODING_PRESETS.keys()))
    preset_data = CODING_PRESETS[selected_preset]
    
    st.info(f"**Problem:** {preset_data['problem']}")
    
    code_input = st.text_area("Python Code Editor", value=preset_data['starter_code'], height=240)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        if st.button("▶️ Execute Code", type="primary", use_container_width=True):
            with st.spinner("Executing script..."):
                exec_result = execute_python_code(code_input)
                st.subheader("Output Console")
                if exec_result["success"]:
                    st.code(exec_result["stdout"] if exec_result["stdout"] else "[No stdout output]", language="text")
                else:
                    st.error("Execution Error:")
                    st.code(exec_result["stderr"], language="text")
                    
    with col_c2:
        if st.button("🧪 Run Automated Test Suite", use_container_width=True):
            with st.spinner("Running test suite..."):
                test_results = run_preset_test_suite(code_input, selected_preset)
                st.subheader("Test Results")
                for tr in test_results:
                    if tr.get("passed"):
                        st.markdown(f'<div class="test-pass">✅ Test #{tr["test_id"]} Passed! Input: <code>{tr["input"]}</code> | Expected: <code>{tr["expected"]}</code></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="test-fail">❌ Test #{tr["test_id"]} Failed! Input: <code>{tr["input"]}</code> | Expected: <code>{tr["expected"]}</code> | Got: <code>{tr.get("actual")}</code></div>', unsafe_allow_html=True)

    with col_c3:
        if st.button("🔍 Run AI Code Review", use_container_width=True):
            with st.spinner("Analyzing Big-O complexity & edge cases..."):
                analysis = analyze_code_with_ai(code_input, problem_description=preset_data["problem"], client=st.session_state.gemini_client)
                st.subheader("AI Feedback")
                st.markdown(f"**Score:** `{analysis.get('correctness_score', 85)}/100`")
                st.markdown(f"**Time Complexity:** `{analysis.get('time_complexity', 'O(n)')}`")
                st.markdown(f"**Space Complexity:** `{analysis.get('space_complexity', 'O(n)')}`")
                st.info(f"**Summary:** {analysis.get('summary', '')}")

# TAB 4: Question Bank & Flashcards
with tab_qbank:
    st.header("📚 Practice Question Bank & Flashcards")
    st.write("Browse technical & behavioral questions, track your confidence levels, and reveal instant model solutions.")
    
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        cat_filter = st.selectbox("Category", ["All", "Software Engineering", "System Design", "Data Science & ML", "Product Management", "Behavioral"])
    with col_q2:
        diff_filter = st.selectbox("Difficulty", ["All", "Easy", "Medium", "Hard"])
    with col_q3:
        search_q = st.text_input("Search Keyword", placeholder="e.g. Hash Map, TinyURL, STAR")
        
    filtered = get_filtered_questions(cat_filter, diff_filter, search_q)
    st.caption(f"Showing {len(filtered)} matched questions")
    
    for q_item in filtered:
        q_id = q_item["id"]
        status_badge = st.session_state.qbank_ratings.get(q_id, "Unrated")
        
        with st.expander(f"[{q_item['difficulty']}] {q_item['title']} - ({q_item['company']}) | Status: {status_badge}"):
            st.markdown(f"**Category:** `{q_item['category']}` > `{q_item['subcategory']}`")
            st.markdown(f"**Question:**\n{q_item['question']}")
            st.markdown(f"💡 **Hint:** {q_item['hint']}")
            
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                if st.button("🟢 Mastered", key=f"mast_{q_id}"):
                    st.session_state.qbank_ratings[q_id] = "🟢 Mastered"
                    st.rerun()
            with col_b2:
                if st.button("🟡 Review Needed", key=f"rev_{q_id}"):
                    st.session_state.qbank_ratings[q_id] = "🟡 Review Needed"
                    st.rerun()
            with col_b3:
                if st.button("🔴 Hard", key=f"hard_{q_id}"):
                    st.session_state.qbank_ratings[q_id] = "🔴 Hard"
                    st.rerun()
            with col_b4:
                if st.button("Reveal Solution", key=f"ans_{q_id}"):
                    st.success(f"**Model Answer:**\n{q_item['model_answer']}")

# TAB 5: Scorecard & Analytics
with tab_analytics:
    st.header("📊 Performance Scorecard & Evaluation Analytics")
    
    if st.session_state.eval_report is None:
        st.info("No active evaluation report found. Complete a mock interview and click 'Generate & View Scorecard' to unlock detailed analytics!")
    else:
        report = st.session_state.eval_report
        
        col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
        with col_s1:
            st.metric("Overall Score", f"{report.get('overall_score', 85)} / 100")
        with col_s2:
            st.metric("Recommendation", report.get("recommendation", "Hire"))
        with col_s3:
            st.markdown("**Assessment Overview:**")
            st.write(report.get("detailed_summary", ""))
            
        st.markdown("---")
        
        col_r_chart, col_r_details = st.columns([1, 1])
        with col_r_chart:
            st.subheader("🕸️ 5-Dimension Competency Radar")
            scores_dict = report.get("scores", {
                "Technical Depth": 85,
                "Problem Solving": 88,
                "Communication": 82,
                "Efficiency": 84,
                "Confidence & Structure": 86
            })
            fig_radar = create_radar_chart(scores_dict)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col_r_details:
            st.subheader("📋 Dimension Score Breakdown")
            df_scores = pd.DataFrame(list(scores_dict.items()), columns=["Competency", "Score (0-100)"])
            st.dataframe(df_scores, use_container_width=True, hide_index=True)
            
            st.subheader("🌟 Key Strengths")
            for str_item in report.get("strengths", []):
                st.markdown(f"✅ {str_item}")
                
            st.subheader("📈 Areas for Improvement")
            for imp_item in report.get("improvements", []):
                st.markdown(f"🎯 {imp_item}")
                
        st.markdown("---")
        
        st.subheader("📥 Export Evaluation Report")
        report_md = f"""# AI Mock Interview Performance Report

**Role:** {session.role}
**Level:** {session.difficulty}
**Overall Score:** {report.get('overall_score', 85)} / 100
**Recommendation:** {report.get('recommendation', 'Hire')}

## Competency Scores
"""
        for k, v in scores_dict.items():
            report_md += f"- **{k}:** {v}/100\n"
            
        report_md += "\n## Key Strengths\n"
        for s in report.get("strengths", []):
            report_md += f"- {s}\n"
            
        report_md += "\n## Areas for Improvement\n"
        for i in report.get("improvements", []):
            report_md += f"- {i}\n"
            
        report_md += f"\n## Detailed Summary\n{report.get('detailed_summary', '')}\n"
        
        st.download_button(
            label="📄 Download Scorecard (Markdown)",
            data=report_md,
            file_name="interview_performance_report.md",
            mime="text/markdown"
        )