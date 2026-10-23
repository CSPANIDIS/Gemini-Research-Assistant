# AI Research Assistant (Gemini Function Calling)

A command-line chat agent that uses Gemini function calling to research topics, analyze and summarize text, and save notes to disk.

Built as the capstone for Google's Gemini for Developers specialization on Coursera.

## Features

The agent exposes four tools to the model:

- **`web_research`**: searches the web with Gemini's built-in Google Search (and optionally reads a given URL) and returns a short 3-bullet summary.
- **`analyze_sentiment`**: classifies a text as positive, negative or neutral, with a one-sentence reason.
- **`generate_concise_summary`**: summarizes a text within an optional word limit (default 100 words).
- **`save_note`**: appends a note to a local text file, creating it if needed.

## How it works

Each user prompt runs through a multi-step agent loop (`handle_user_turn` in `agent.py`):

1. The model receives the conversation history along with the tool declarations.
2. If it requests one or more tools, the dispatcher (`run_custom_function`) executes them locally.
3. The results are sent back to the model as function responses.
4. Steps 1-3 repeat until the model replies with plain text, which is the final answer.

The loop is capped at 5 rounds per prompt. If no final answer is reached, the agent stops and asks you to try a simpler request. Conversation history is kept across prompts within a session.

## Setup

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your key:

```
GEMINI_API_KEY=your_key_here
```

Then run:

```bash
python agent.py
```

Type `exit` or `quit` to leave the chat.

## Example prompts

- `What are the latest developments in solid-state batteries?`
- `Analyze the sentiment of this review: "The battery life is great but the screen scratches easily."`
- `Summarize this article in 50 words: <paste text>`
- `Research recent progress in quantum error correction and save a note to notes.txt`

The last prompt chains tools: the agent calls `web_research`, then `save_note` with the findings.

## What I changed from the course template

- Implemented `analyze_sentiment` and `generate_concise_summary` and their schemas (the course left them as TODOs).
- Fixed the dispatcher: argument names didn't match the schemas, the optional `max_length` was not handled, and results were double-wrapped so errors appeared as "ok".
- Fixed declaration order (`NameError`: schemas were used before they were defined).
- Replaced the course's fixed two-turn flow with a multi-step agent loop, so the agent can chain tools (e.g. research, then save a note).

## Project structure

- `agent.py`: tool implementations, function declarations, dispatcher, agent loop and CLI entry point.
- `requirements.txt`: Python dependencies (`google-genai`, `python-dotenv`).
- `.env.example`: template for the `GEMINI_API_KEY` environment variable.
- `.gitignore`: files excluded from version control.
- `.github/workflows/`: GitHub Actions workflows for Claude Code.
- `README.md`: this file.
