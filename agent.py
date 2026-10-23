"""Gemini-powered research assistant.

Exposes a small set of custom tools (note saving, web research, sentiment
analysis and summarization) to a Gemini model through manual function calling,
and runs an interactive command-line conversation loop on top of them.

Requires the ``GEMINI_API_KEY`` environment variable (a ``.env`` file is
supported).
"""

import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Setup & client ----------------------------------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=api_key)
MODEL_ID = "gemini-3.8-flash"  # tools-capable model


# --- 1. Custom function implementations --------------------------------------
def save_note(content: str, filename: str) -> str:
    """Append a note to a local text file.

    Args:
        content: Text of the note to store.
        filename: Path of the file to append to; created if it does not exist.

    Returns:
        A confirmation message on success, or an error message describing the
        failure.
    """
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(content + "\n\n")
        return f"Note saved to '{filename}'."
    except Exception as e:
        return f"Error saving note: {e!r}"

def web_research(query: str, url: str | None = None) -> Dict[str, Any]:
    """Research a topic on the web and return a short summary.

    Uses Gemini's built-in ``google_search`` and ``url_context`` tools to
    gather up-to-date information.

    Args:
        query: Topic or question to research.
        url: Optional URL to read and use as additional context.

    Returns:
        ``{"status": "ok", "summary": <text>}`` on success, or
        ``{"status": "error", "error": <message>}`` if the request fails.
    """
    tools = [
        {"google_search": {}},
        {"url_context": {}},
    ]

    prompt_parts = [
        f"Use Google Search to find recent, relevant information about: {query}.",
        "Provide a short 3-bullet summary with one sentence per bullet.",
    ]
    if url:
        prompt_parts.append(
            f"Also read and use this URL as context if relevant: {url}"
        )

    prompt = " ".join(prompt_parts)

    try:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(tools=tools),
        )
        summary_text = resp.text or ""
        return {"status": "ok", "summary": summary_text}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def analyze_sentiment(analyzed_text:str) -> Dict[str, Any]:
    """Classify the sentiment of a text.

    Args:
        analyzed_text: The text to analyze.

    Returns:
        ``{"status": "ok", "result": <text>}`` where the result classifies the
        text as positive, negative or neutral with a one-sentence reason, or
        ``{"status": "error", "error": <message>}`` if the request fails.
    """

    prompt = ('Analyze the following text and classify it as positive, negative or neutral ' 
                f'and give a one sentence reason:\n{analyzed_text}')

    try:
        resp = client.models.generate_content(
            model= MODEL_ID,        
            contents= prompt,     
        )
        
        sentiment_result = resp.text or ""

        return {"status": "ok", "result": sentiment_result}  
    except Exception as e:
        return {"status": "error", "error": str(e)}  




def generate_concise_summary(text: str, max_length: int = 100) -> Dict[str, Any]:
    """Summarize a long text within a word limit.

    Args:
        text: The text to summarize.
        max_length: Maximum number of words the summary may contain.

    Returns:
        ``{"status": "ok", "summary": <text>}`` on success, or
        ``{"status": "error", "error": <message>}`` if the request fails.
    """
    prompt = (
        f"Create a concise summary of the provided text that can be up to {max_length} words\n"
        f"Here's the text: \n{text}"
    )

    try:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )

        summary_result = resp.text or ""

        return {"status":"ok", "summary": summary_result}

    except Exception as e:
        return {"status": "error", "error": str(e)}




# --- 2. Function declarations (schemas) --------------------------------------
# Each declaration describes a custom function to the model so it can decide
# when to call it and with which arguments.

# Schema for save_note()
save_note_decl = types.FunctionDeclaration(
    name="save_note",
    description="Save a short research note to a local text file.",
    parameters_json_schema={
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Text of the note."},
            "filename": {
                "type": "string",
                "description": "File name, e.g. 'research_notes.txt'.",
            },
        },
        "required": ["content", "filename"],
    },
)

# Schema for web_research()
web_research_decl = types.FunctionDeclaration(
    name="web_research",
    description=(
        "Search the web for recent information using Google Search and "
        "optionally analyze a specific URL, then return a short summary."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic or question to research.",
            },
            "url": {
                "type": "string",
                "description": "Optional URL to read/analyze.",
            },
        },
        "required": ["query"],
    },
)


# Schema for analyze_sentiment()
analyze_sentiment_decl = types.FunctionDeclaration(
    name="analyze_sentiment",
    description=(
        "Classify the sentiment of a text as positive, negative or neutral, "
        "with a one-sentence reason."
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "analyzed_text": {"type": "string", "description": "The text to analyze."},
        },
        "required": ["analyzed_text"],
    },
)

# Schema for generate_concise_summary()
generate_concise_summary_decl = types.FunctionDeclaration(
    name="generate_concise_summary",
    description=(
        "Generate a concise summary of a long text, with an optional word limit"
    ),
    parameters_json_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The text to summarize"},
            "max_length": {"type": "integer", "description": "The maximum number of words that the summarized text can have, defaults to 100"},
        },
        "required": ["text"]
    },
)


# Bundle all declarations into a single tool exposed to the model.
custom_functions_tool = types.Tool(
    function_declarations=[
        save_note_decl,
        analyze_sentiment_decl,
        generate_concise_summary_decl,
        web_research_decl,
    ]
)

TOOLS = [custom_functions_tool]



# --- Function-call dispatcher -------------------------------------------------
def run_custom_function(fc: types.FunctionCall | Any) -> Dict[str, Any]:
    """Dispatch a model-requested function call to its implementation.

    Args:
        fc: The function call returned by the model, or a part wrapping one.

    Returns:
        The called function's result dictionary, or
        ``{"status": "error", "error": <message>}`` if the function name is
        unknown or the call raises.
    """
    name = getattr(fc, "name", None)
    call_obj = getattr(fc, "function_call", fc)
    args = dict(getattr(call_obj, "args", {}) or {})

    try:
        if name == "save_note":
            msg = save_note(args["content"], args["filename"])
            return {"status": "ok", "message": msg}

        if name == "analyze_sentiment":
            return analyze_sentiment(
                analyzed_text=args["analyzed_text"])

        if name == "generate_concise_summary":
            return generate_concise_summary(
                text=args["text"], max_length=int(args.get("max_length", 100))
            )
            

        if name == "web_research":
            return web_research(
                query=args["query"],
                url=args.get("url"),
            )

        return {"status": "error", "error": f"Unknown function '{name}'."}

    except Exception as e:
        return {"status": "error", "error": str(e)}


# --- Response text helper ----------------------------------------------------
def extract_text(response: types.GenerateContentResponse) -> str:
    """Return the text of a model response.

    Prefers the response's ``text`` shortcut and otherwise concatenates the
    text parts of all candidates, joined by newlines.

    Args:
        response: The response returned by ``generate_content``.

    Returns:
        The extracted text, or an empty string if the response has none.
    """
    if getattr(response, "text", None):
        return response.text
    pieces: List[str] = []
    for cand in response.candidates or []:
        if cand.content and cand.content.parts:
            for part in cand.content.parts:
                if getattr(part, "text", None):
                    pieces.append(part.text)
    return "\n".join(pieces).strip()


# --- Two-turn function-calling per user prompt -------------------------------
def handle_user_turn(history: List[types.Content], user_prompt: str) -> str:
    """Process one user prompt using manual two-turn function calling.

    The first model call may request custom functions; these are executed
    locally and their results are sent back in a second call that produces the
    final answer. ``history`` is updated in place with every message exchanged.

    Args:
        history: Conversation so far; mutated by this function.
        user_prompt: The user's latest message.

    Returns:
        The assistant's final answer, or an error message if a model call
        fails.
    """
    # 1) Record the user message
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_prompt)],
    )
    history.append(user_content)

    # First turn: model decides which function(s) to call
    try:
        first = client.models.generate_content(
            model=MODEL_ID,
            contents=history,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True  # we manage function calling manually
                ),
            ),
        )
    except Exception as e:
        return f"Error calling the model: {e!r}"

    history.append(first.candidates[0].content)
    function_calls = first.function_calls or []

    # Model answered directly (or only used its own tools inside web_research)
    if not function_calls:
        return extract_text(first)

    # 2) Execute requested custom functions and send results back
    # Collect one function-response part per requested call
    response_parts = []
    for fc in function_calls:
        result = run_custom_function(fc)
        response_parts.append(types.Part.from_function_response(name=fc.name, response=result))
    history.append(types.Content(role="user", parts=response_parts))

    # Second turn: the model composes the final answer from the function
    # results; tools are disabled so it cannot request further calls.
    try:
        second = client.models.generate_content(
            model=MODEL_ID,
            contents=history,
            config=types.GenerateContentConfig(
                tools=[],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    except Exception as e:
        return f"Error generating final answer: {e!r}"
    history.append(second.candidates[0].content)
    return extract_text(second)


# --- Conversation loop Execution ---------------------------------------------
def main() -> None:
    """Run the interactive assistant until the user types 'exit' or 'quit'."""
    print("AI Research Assistant ready.")
    print("Type 'exit' to quit.\n")
    history: List[types.Content] = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        answer = handle_user_turn(history, user_input)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()