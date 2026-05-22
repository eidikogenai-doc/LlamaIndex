# Day 7 — Chat Engine — Memory & Context

## What this script does
Implements a full conversational RAG chatbot with memory. Covers all chat
engine modes, demonstrates how follow-up questions work with memory,
compares all modes side by side, manages conversation history with token
limits, and implements streaming chat for better user experience.

## Concepts Covered
- ChatEngine = QueryEngine + conversation memory
- Why QueryEngine fails on follow-up questions
- condense_question mode — condenses history + query into standalone question
- context mode — retrieves fresh context for every message
- condense_plus_context mode — best for production RAG (recommended)
- simple mode — pure LLM chat, no document retrieval
- react mode — agent-style, LLM decides whether to retrieve
- ChatMemoryBuffer — stores conversation history with token limit
- token_limit — prevents context window overflow
- ChatMessage and MessageRole — SYSTEM, USER, ASSISTANT
- system_prompt — sets LLM persona and behavior
- chat() — synchronous, waits for full response
- stream_chat() — streams tokens as generated
- achat() — async version for web servers
- chat_engine.reset() — clears conversation history
- chat_engine.chat_history — inspect stored messages
- verbose=True — shows internal reasoning steps

## Real-World Use Case
**HR Policy Chatbot for Employees**
An HR department deploys a chatbot over their employee handbook.
condense_plus_context ensures employees can ask "How many days off do I get?"
followed by "What about for sick leave?" — the "what about" follow-up
works because memory condenses it to a standalone question. system_prompt
restricts the bot to HR topics only. ChatMemoryBuffer(token_limit=3000)
prevents the conversation from exceeding the LLM context window during
long sessions. stream_chat() gives employees immediate feedback instead
of waiting 5-10 seconds for a complete response.

## How to Run
```bash
python day7.py
# Option 1: Load documents
# Option 2: Build index
# Option 5: condense_plus_context (recommended — start here)
# Option 8: Mode comparison (most educational)
# Option 6: Memory management — watch history truncate
# Option 9: Streaming chat
```

## Testing Tips
- Ask a question, then follow up with "When was it formed?" or "Tell me more about that"
- Type 'history' to see stored messages
- Type 'reset' to clear memory and test again
- Try the same two questions across all modes in Demo 8

## Dependencies
```
llama-index-core
llama-index-embeddings-huggingface
sentence-transformers
pypdf
```