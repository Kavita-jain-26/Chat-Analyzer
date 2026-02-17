# Chat-Analyzer
Manage stateless LLM conversations with automatic token tracking and cost analysis.
# Conversation Analyzer

A simple tool to track and analyze multi-turn conversations with LLMs. Built to understand token usage, costs, and conversation patterns.

## What it does

- Tracks tokens for each message in a conversation
- Calculates estimated API costs
- Maintains conversation history (for stateless LLM calls)
- Generates insights about token usage patterns

## Installation

pip install tiktoken openai python-dotenvSet up your `.env` file with your OpenAI API key.

## Quick example

from conversation_analyzer import ConversationAnalyzer

analyzer = ConversationAnalyzer()
analyzer.start_conversation()

# Chat and it automatically tracks everything
response = analyzer.chat_with_analysis("Hi, I'm John")
print(response)

# Ask follow-up - it remembers context
response2 = analyzer.chat_with_analysis("What's my name?")
print(response2)  # "Your name is John"

# See the analysis
insights = analyzer.generate_insights()
print(insights)## Main methods

- `start_conversation()` - Start tracking a new conversation
- `chat_with_analysis(message)` - Chat with OpenAI and track automatically
- `analyze_conversation()` - Get stats (tokens, costs, trends)
- `generate_insights()` - Get a readable report
- `print_conversation_flow()` - Visual view of the conversation

## Why I built this

This was part of learning about stateless LLM calls and token management. Every API call is independent, so you need to pass the full conversation history each time. This tool handles that while tracking token usage and costs.

## Requirements

- Python 3.8+
- tiktoken, openai, python-dotenv
