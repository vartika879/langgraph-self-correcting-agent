import os
import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel
import operator

load_dotenv()

st.set_page_config(page_title="AI Content Writer & Validator", page_icon="✍️", layout="wide")

st.title("✍️ AI Content Generator with Self-Correction")
st.caption("LangGraph + Groq LLM streaming with iterative validation loop")

# Fetch API key directly from environment (.env)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found! Please check your .env file.")
    st.stop()

# Sidebar Configuration (Without API Key Input)
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Model", 
        ["openai/gpt-oss-20b"]
    )
    max_retries = st.slider("Max Validation Retries", min_value=1, max_value=5, value=3)

# LLM Initialization
llm = ChatGroq(api_key=api_key, model=model_name, temperature=0, reasoning_effort="low")

# State Definition
class AgentState(TypedDict):
    task: str
    content_draft: str
    validation_error: list[str]
    retry_count: int
    max_retries: int
    is_valid: bool
    history: Annotated[list[dict], operator.add]

# Graph Nodes
def generator_node(state):
    task = state["task"]
    content_draft = state.get("content_draft", "")
    errors = state.get("validation_error", [])
    retry_count = state.get("retry_count", 0)

    if content_draft:
        prompt = f"""You are a content writer. You previously wrote a draft for this task, but it had issues.
Original Task:
{task}

Previous draft:
{content_draft}

Issue found in the previous draft:
{errors}
Rewrite the content to fix ALL the issues listed above.
Keep everything that was already correct — only fix what's flagged.
Do not introduce new problems while fixing these issues."""
        label = f"🔄 Revising Draft (Attempt #{retry_count + 1})"
    else:
        prompt = f"""You are a content writer. Write content based on the following task.

Task:
{task}

Follow the instructions in the task carefully, including any length, tone, or format requirements."""
        label = "📝 Generating Initial Draft"

    with st.status(label, expanded=True) as status_box:
        draft_placeholder = st.empty()
        full_response = ""
        for chunk in llm.stream(prompt):
            full_response += chunk.content
            draft_placeholder.markdown(full_response + "▌")
        
        draft_placeholder.markdown(full_response)
        status_box.update(label=f"{label} - Completed", state="complete", expanded=False)

    return {
        "content_draft": full_response,
        "history": [{"node": "generator_node", "content": full_response}]
    }

class Validation(BaseModel):
    is_valid: bool
    validation_error: list[str]

def validate_content_node(state):
    content_draft = state.get("content_draft", "")
    retry_count = state.get("retry_count", 0)
    task = state["task"]

    prompt = f"""You are a strict content validator.

Original Task:
{task}

Content Draft:
{content_draft}

Validate the draft against ALL requirements in the original task.
Check:
- Exact word count if specified
- Required number of bullet points
- Tone
- Topic/relevance
- Formatting
- Every explicit requirement in the task

Be strict. If even one explicit requirement is not satisfied, return is_valid=False and clearly explain the issue.
If all requirements are satisfied, return is_valid=True and an empty validation_error list."""

    with st.status("🔍 Validating Content Quality...", expanded=True) as status_box:
        structured_llm = llm.with_structured_output(Validation, method="json_schema")
        response = structured_llm.invoke(prompt)

        if response.is_valid:
            st.success("✅ Content passed all quality and format checks!")
            status_box.update(label="Validation Passed", state="complete", expanded=False)
        else:
            st.error(f"❌ Validation Failed with {len(response.validation_error)} issue(s):")
            for err in response.validation_error:
                st.write(f"- {err}")
            status_box.update(label="Validation Failed", state="error", expanded=True)

    result = {
        "is_valid": response.is_valid,
        "validation_error": response.validation_error,
        "history": [{"node": "validate_content_node", "content": response.validation_error}]
    }
    if not response.is_valid:
        result["retry_count"] = retry_count + 1

    return result

def fallback_node(state):
    content_draft = state.get("content_draft", "")
    validation_error = state.get("validation_error", [])
    st.warning("⚠️ Max retry limit reached. Returning closest possible draft.")

    warning = f"\n\n> ⚠️ **Note:** Unresolved validation issues: {validation_error}"
    return {
        "content_draft": content_draft + warning,
        "history": [{"node": "fallback_node", "content": content_draft + warning}]
    }

def route_by_status(state: AgentState) -> str:
    if state.get("is_valid", False):
        return END
    elif state.get("retry_count", 0) < state.get("max_retries", 3):
        return "generator_node"
    else:
        return "fallback_node"

# Graph Construction
graph = StateGraph(AgentState)
graph.add_node("generator_node", generator_node)
graph.add_node("validate_content_node", validate_content_node)
graph.add_node("fallback_node", fallback_node)

graph.add_edge(START, "generator_node")
graph.add_edge("generator_node", "validate_content_node")
graph.add_conditional_edges("validate_content_node", route_by_status, {
    "generator_node": "generator_node",
    "fallback_node": "fallback_node",
    END: END
})
graph.add_edge("fallback_node", END)
app = graph.compile()

# UI Input & Execution
task_input = st.text_area(
    "Enter your prompt/task:",
    placeholder="e.g. Write a 3-bullet summary of Quantum Computing in a playful tone.",
    height=100
)

if st.button("Generate & Validate Content", type="primary", use_container_width=True):
    if not task_input.strip():
        st.warning("Please enter a task before generating.")
    else:
        initial_state = {
            "task": task_input,
            "content_draft": "",
            "validation_error": [],
            "retry_count": 0,
            "max_retries": max_retries,
            "is_valid": False,
            "history": []
        }

        st.divider()
        final_state = app.invoke(initial_state)

        st.subheader("🎯 Final Result")
        st.markdown(final_state["content_draft"])