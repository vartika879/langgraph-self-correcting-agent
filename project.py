import os
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
from typing import TypedDict,Annotated
from pydantic import BaseModel
from langchain_core.messages import BaseMessage
import operator


load_dotenv()


api=os.getenv("GROQ_API_KEY")

llm=ChatGroq(api_key=api,model="openai/gpt-oss-20b",
             temperature=0,reasoning_effort="low")

class AgentState(TypedDict):
    task:str
    content_draft:str
    validation_error:list[str]
    retry_count:int
    max_retries:int
    is_valid:bool
    history: Annotated[list[dict], operator.add]


def generator_node(state):
    """
    Generates content based on the task in state.

    First call: writes fresh content from the task instructions.
    Retry call: revises the previous draft using validation_errors as feedback.

    Returns:
        dict: {"content_draft": response.content}
    """

    task=state["task"]
    content_draft=state.get("content_draft","")
    errors=state.get("validation_error","")

    if content_draft:
        prompt=f"""You are a content writer . you previously wrote a draft for this task , but it had issues
        Original Task
        {task}
        
        Previous draft:
        {content_draft}

        Issue found in the previous draft:
        {errors}
Rewrite the content to fix ALL the issues listed above.
Keep everything that was already correct — only fix what's flagged.
Do not introduce new problems while fixing these issues.

"""
    else: 
        prompt =f"""
You are a content writer. Write content based on the following task.

Task:
{task}

Follow the instructions in the task carefully, including any length, tone, or format requirements."""
    full_response = ""
    for chunk in llm.stream(prompt):
        text = chunk.content
        print(text, end="", flush=True)  # flush=True se bina delay ke turant print hota hai
        full_response += text
        
    print("\n")

    return {
        "content_draft": full_response,
        "history": [{"node": "generator_node", "content": full_response}]
    }
      



class Validation(BaseModel):
    is_valid:bool
    
    validation_error:list[str]



def validate_content_node(state):
    """
    Validates the current content draft against quality criteria.

    Checks the draft for issues like length (too long/short), tone mismatch,
    irrelevant content, and formatting problems.

    Returns:
        dict: {"is_valid": bool, "validation_errors": list of found issues}
    """
    content_draft=state.get("content_draft","")
    retry_count = state.get("retry_count", 0)
    task=state["task"]

    

    prompt = f"""
You are a strict content validator.

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

Be strict. If even one explicit requirement is not satisfied,
return is_valid=False and clearly explain the issue.

If all requirements are satisfied, return is_valid=True
and an empty validation_error list.
"""
    
    structured_llm=llm.with_structured_output(Validation,method="json_schema")
    response=structured_llm.invoke(prompt)


   

    result={
        "is_valid":response.is_valid,
       
        "validation_error":response.validation_error,
        "history": [{"node": "validate_content_node", "content": response.validation_error}]
    }
    if not response.is_valid:
        result["retry_count"] = retry_count + 1

    return result



def fallback_node(state):
    """
    Called when max retries are exhausted and content is still invalid.
    Returns the best-available draft with a warning, instead of nothing.
    """
    content_draft = state.get("content_draft", "")
    validation_error = state.get("validation_error", [])

    warning = (
        f"\n\n[Note: This content could not be fully validated after "
        f"multiple attempts. Remaining issues: {validation_error}]"
    )

    return {
        "content_draft": content_draft + warning,
        "history": [{"node": "fallback_node", "content": content_draft + warning}]
    }



def route_by_status(state:AgentState) -> str:
    is_valid = state.get("is_valid", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 5)
    
    

    if is_valid:
        return END
    elif retry_count < max_retries:
        return "generator_node"
    else:
         return "fallback_node"



graph=StateGraph(AgentState)

graph.add_node("generator_node",generator_node)
graph.add_node("validate_content_node",validate_content_node)
graph.add_node("fallback_node",fallback_node)


graph.add_edge(START,"generator_node")
graph.add_edge("generator_node","validate_content_node")

graph.add_conditional_edges("validate_content_node",route_by_status,{
        "generator_node": "generator_node",
        "fallback_node": "fallback_node",
         END: END
    })



graph.add_edge("fallback_node",END)
app=graph.compile()

if __name__ == "__main__":
    initial_state = {
        "task": "Write a short 3-point summary on Artificial Intelligence.",
        "content_draft": "",
        "validation_error": [],
        "retry_count": 0,
        "max_retries": 3,
        "is_valid": False,
        "history": []
    }

    final_output = app.invoke(initial_state)
    
    print("\n--- Execution Completed ---")
    print(f"Valid: {final_output['is_valid']}")
    print(f"Final Retries: {final_output['retry_count']}")