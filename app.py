import os
import streamlit as st

from project import app

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Content Validator & Fixer",
    page_icon="✅",
    layout="centered",
)

# ------------------------------------------------------------------
# STYLES
# ------------------------------------------------------------------
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 800; margin-bottom: 0; }
    .subtitle { color: #888; margin-top: 0.2rem; margin-bottom: 1.2rem; }
    .final-content {
        border: 1px solid #444;
        border-radius: 14px;
        padding: 20px;
        background: rgba(255,255,255,0.03);
    }
    .attempt-card {
        border: 1px solid #333;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.03);
    }
    .pipeline-track {
        display: flex;
        gap: 6px;
        margin-bottom: 1rem;
    }
    .pipeline-step {
        flex: 1;
        text-align: center;
        font-size: 0.75rem;
        padding: 6px 4px;
        border-radius: 8px;
        background: rgba(255,255,255,0.05);
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# EXAMPLE TASK — same as project.py's __main__ block
# ------------------------------------------------------------------
EXAMPLE_TASK = "Write a short 3-point summary on Artificial Intelligence."

if "task_value" not in st.session_state:
    st.session_state.task_value = ""

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.markdown('<div class="main-title">✅ Content Validator & Fixer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Give it a task, it writes, checks its own work against every requirement, and retries until it passes — or hands you its best attempt.</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="pipeline-track">
  <div class="pipeline-step">1️⃣ Generate</div>
  <div class="pipeline-step">2️⃣ Validate</div>
  <div class="pipeline-step">3️⃣ Retry (loop)</div>
  <div class="pipeline-step">4️⃣ Fallback / Done</div>
</div>
""", unsafe_allow_html=True)

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not found in your .env file. project.py needs it to build the LLM.")
    st.stop()

st.button(
    "📋 Load example task",
    on_click=lambda: st.session_state.update(task_value=EXAMPLE_TASK),
    use_container_width=True,
)

# ------------------------------------------------------------------
# MAIN FORM
# ------------------------------------------------------------------
with st.form("pipeline_form"):
    task = st.text_area(
        "Task",
        value=st.session_state.task_value,
        height=180,
        placeholder="Describe what to write and every requirement it must satisfy...",
    )

    max_retries = st.number_input("Max retries", min_value=1, max_value=6, value=3)

    submitted = st.form_submit_button("🚀 Run pipeline", use_container_width=True, type="primary")

st.session_state.task_value = task

# ------------------------------------------------------------------
# RUN PIPELINE — calls app.stream() from project.py directly
# ------------------------------------------------------------------
if submitted:
    if not task.strip():
        st.error("Please describe the task first.")
        st.stop()

    initial_state = {
        "task": task,
        "content_draft": "",
        "validation_error": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "is_valid": False,
        "history": [],
    }

    progress_box = st.status("Running pipeline...", expanded=True)
    attempt_num = 0
    final_state = {}

    try:
        for step_output in app.stream(initial_state):
            node_name = list(step_output.keys())[0]
            data = step_output[node_name]
            final_state.update(data)

            if node_name == "generator_node":
                attempt_num += 1
                preview = data.get("content_draft", "")[:120].replace("\n", " ")
                progress_box.write(f"**Attempt {attempt_num} — ✍️ generator_node** — {preview}...")

            elif node_name == "validate_content_node":
                if data.get("is_valid"):
                    progress_box.write(f"**Attempt {attempt_num} — ✅ validate_content_node** — passed")
                else:
                    issues = data.get("validation_error", [])
                    progress_box.write(f"**Attempt {attempt_num} — ❌ validate_content_node** — {', '.join(issues)}")

            elif node_name == "fallback_node":
                progress_box.write("**⚠️ fallback_node** — retries exhausted, releasing best attempt")

        progress_box.update(label="Pipeline complete ✅", state="complete", expanded=False)

        tab1, tab2 = st.tabs(["✨ Final Content", "🧩 Attempt History"])

        with tab1:
            st.markdown('<div class="final-content">', unsafe_allow_html=True)
            st.markdown(final_state.get("content_draft", "_No content generated._"))
            st.markdown('</div>', unsafe_allow_html=True)

            st.write("")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Valid", str(final_state.get("is_valid", False)))
            with col2:
                st.metric("Retries used", f"{final_state.get('retry_count', 0)} / {max_retries}")

            st.download_button(
                "⬇️ Download content (.txt)",
                data=final_state.get("content_draft", ""),
                file_name="final_content.txt",
                use_container_width=True,
            )

        with tab2:
            history = final_state.get("history", [])
            if not history:
                st.write("No history recorded.")
            for entry in history:
                node = entry.get("node", "unknown")
                content = entry.get("content", "")
                icon = {"generator_node": "✍️", "validate_content_node": "🔎", "fallback_node": "⚠️"}.get(node, "•")
                st.markdown(f'<div class="attempt-card"><b>{icon} {node}</b><br>{content}</div>', unsafe_allow_html=True)

    except Exception as e:
        progress_box.update(label="Pipeline failed ❌", state="error")
        st.error(f"Something went wrong: {e}")