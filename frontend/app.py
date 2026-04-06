import streamlit as st
import requests
import time
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="AI Job Preparation Assistant", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")

# Load Backend URL correctly whether local or deployed
if "BACKEND_URL" in st.secrets:
    BASE_URL = st.secrets["BACKEND_URL"]
else:
    BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.markdown("""
    <style>
    /* Modern, theme-adaptive cards */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    /* Dynamic colors based on score */
    .score-high { color: #2e7d32; }   /* Green for light mode */
    .score-medium { color: #f57c00; } /* Orange */
    .score-low { color: #d32f2f; }    /* Red */
    
    /* Adapt for dark theme automatically */
    @media (prefers-color-scheme: dark) {
        .score-high { color: #81c784; }
        .score-medium { color: #ffb74d; }
        .score-low { color: #e57373; }
    }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'results' not in st.session_state:
    st.session_state.results = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    base_url_input = st.text_input("Backend URL", value=BASE_URL)
    if base_url_input:
        BASE_URL = base_url_input.rstrip("/")
        
    st.markdown("---")
    use_sample = st.toggle("🧪 Demo Mode (Use Sample Data)")
    
    st.markdown("---")
    st.markdown("### 🌓 Theme settings")
    st.info("To toggle **Light / Dark mode**, click the **⋮** menu in the top right corner of the page, select **Settings**, and change the **Theme**.")

# --- HEADER ---
st.title("🎯 AI Job Preparation Assistant")
st.markdown("Upload your CV and paste the Job Description to get a comprehensive match analysis, personalized study syllabus, and tailored interview questions.")

# --- MAIN INPUT SECTION ---
with st.container():
    st.subheader("📝 1. Input Details")
    col1, col2 = st.columns(2)
    
    with col1:
        cv_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"], disabled=use_sample, help="Upload your latest CV in PDF format.")
        if use_sample:
            st.warning("Demo Mode ACTIVE. CV upload is bypassed.")
            
    with col2:
        jd_default = "We are looking for a Senior React Developer with 5+ years of experience in JavaScript, React, Redux, Node.js, and AWS. Must have experience with microservices architecture and CI/CD pipelines." if use_sample else ""
        jd_text = st.text_area("Job Description (JD)", height=150, value=jd_default, placeholder="Paste the job description here...")

# --- PROCESSING BLOCK ---
def run_analysis(cv_bytes, jd_content):
    try:
        progress_bar = st.progress(0, text="Initializing analysis...")
        
        # Step 1
        progress_bar.progress(10, text="📄 Analyzing Job Description...")
        jd_res = requests.post(f"{BASE_URL}/analyze/jd", json={"job_description": jd_content})
        jd_res.raise_for_status()
        jd_data = jd_res.json()
        
        # Step 2
        progress_bar.progress(30, text="🔍 Analyzing CV...")
        if use_sample:
            cv_data = {
                "skills": ["JavaScript", "React", "HTML", "CSS", "Node.js", "Git"],
                "experience_summary": "Frontend developer with 3 years of experience.",
                "domains": ["Web Development"]
            }
            time.sleep(1)
        else:
            cv_res = requests.post(f"{BASE_URL}/analyze/cv", files={"file": ("cv.pdf", cv_bytes, "application/pdf")})
            cv_res.raise_for_status()
            cv_data = cv_res.json()

        # Step 3
        progress_bar.progress(60, text="⚖️ Calculating Match Score...")
        match_payload = {"jd_skills": jd_data.get("skills", []), "cv_skills": cv_data.get("skills", []), "cv_text": None}
        match_res = requests.post(f"{BASE_URL}/match", json=match_payload)
        match_res.raise_for_status()
        match_data = match_res.json()
        
        missing_skills = match_data.get("missing_skills", [])
        
        # Step 4
        progress_bar.progress(80, text="📚 Generating Study Syllabus...")
        syllabus_data = []
        if missing_skills:
            syll_res = requests.post(f"{BASE_URL}/syllabus", json={"missing_skills": missing_skills})
            try:
                syll_res.raise_for_status()
                syllabus_data = syll_res.json()
            except requests.exceptions.RequestException:
                pass

        # Step 5
        progress_bar.progress(95, text="🎙️ Preparing Interview Questions...")
        interview_data = {}
        int_res = requests.post(f"{BASE_URL}/interview", json={"job_description": jd_content, "missing_skills": missing_skills})
        try:
            int_res.raise_for_status()
            interview_data = int_res.json()
        except requests.exceptions.RequestException:
            pass

        progress_bar.progress(100, text="✅ Analysis Complete!")
        time.sleep(0.5)
        progress_bar.empty()

        return {"jd": jd_data, "cv": cv_data, "match": match_data, "syllabus": syllabus_data, "interview": interview_data}

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}\n\nMake sure the backend is running at {BASE_URL}.")
        return None

# --- ACTION BUTTON ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Analyze My Compatibility", use_container_width=True, type="primary"):
    if not jd_text.strip():
        st.error("Please paste the Job Description.")
    elif not use_sample and cv_file is None:
        st.error("Please upload a CV.")
    else:
        cv_bytes = None if use_sample else cv_file.getvalue()
        results = run_analysis(cv_bytes, jd_text)
        if results:
            st.session_state.results = results
            st.rerun()

# --- RESULTS SECTION ---
if st.session_state.results:
    st.markdown("---")
    res = st.session_state.results
    match_data = res["match"]
    
    score_pct = match_data.get('match_percentage', 0)
    
    # Select color class based on score
    if score_pct >= 75:
        score_class = "score-high"
        msg = "Excellent Match!"
    elif score_pct >= 50:
        score_class = "score-medium"
        msg = "Good Potential. Needs some work."
    else:
        score_class = "score-low"
        msg = "Significant Skills Gap."

    # Top Level Overview Callout
    st.markdown(f'''
    <div class="metric-card">
        <div style="font-size: 1.2rem; color: var(--text-color); opacity: 0.8;">Overall Match Percentage</div>
        <div class="metric-value {score_class}">{score_pct:.1f}%</div>
        <div style="font-size: 1.1rem; font-weight: 500;" class="{score_class}">{msg}</div>
    </div>
    <br><br>
    ''', unsafe_allow_html=True)
    
    # Use tabs for a clean, organized UI
    tab1, tab2, tab3 = st.tabs(["📊 Skills Analysis", "📚 Study Syllabus", "🎯 Interview Prep"])
    
    with tab1:
        st.subheader("Skills Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.success("✅ **Strengths (Matching Skills)**")
            strengths = match_data.get('strengths', [])
            if strengths:
                for s in strengths:
                    st.markdown(f"- {s}")
            else:
                st.markdown("*None identified*")
                
        with c2:
            st.error("❌ **Missing Skills (Gap Analysis)**")
            missing_skills = match_data.get('missing_skills', [])
            if missing_skills:
                for m in missing_skills:
                    st.markdown(f"- {m}")
            else:
                st.markdown("*You meet all the required skills!*")

    with tab2:
        st.subheader("Customized Learning Path")
        syllabus_data = res.get("syllabus", [])
        if syllabus_data and isinstance(syllabus_data, list):
            for idx, item in enumerate(syllabus_data):
                with st.expander(f"📖 Module {idx+1}: {item.get('topic', 'Topic')} (Difficulty: {item.get('difficulty', 'Unknown')})"):
                    st.markdown("**Subtopics:**")
                    for sub in item.get('subtopics', []):
                        st.markdown(f"- {sub}")
                    st.markdown("**Practice Questions:**")
                    for q in item.get('practice_questions', []):
                        st.markdown(f"- {q}")
        else:
            st.info("🎉 No study syllabus needed based on your current match!")

    with tab3:
        st.subheader("Anticipated Interview Questions")
        interview_data = res.get("interview", {})
        questions = interview_data.get("questions", [])
        if questions:
            for i, q in enumerate(questions, 1):
                st.markdown(f"**Q{i}:** {q}")
        else:
            st.info("No customized interview questions generated.")
            
    st.markdown("---")
    
    # DOWNLOAD BUTTON
    report_text = f"--- AI JOB PREPARATION REPORT ---\nMatch Score: {score_pct}%\n\nSTRENGTHS:\n"
    report_text += "\n".join([f"- {s}" for s in match_data.get('strengths', [])]) if match_data.get('strengths', []) else "None identified"
    report_text += "\n\nMISSING SKILLS:\n"
    report_text += "\n".join([f"- {m}" for m in missing_skills]) if missing_skills else "None identified"
    
    col_dl, _ = st.columns([1, 2])
    with col_dl:
        st.download_button("📥 Download Report Summary", data=report_text, file_name="application_report.txt", mime="text/plain")
