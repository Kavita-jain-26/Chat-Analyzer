"""
Multi-turn Conversation Analyzer
Demonstrates Day 4 concepts: tokenization, stateless LLM calls, conversation management

This module provides tools to:
- Track multi-turn conversations
- Count tokens per message
- Analyze conversation patterns
- Generate insights about token usage
- Calculate costs
"""

import tiktoken
from openai import OpenAI
from datetime import datetime
from typing import Optional, List, Dict, Any


class ConversationAnalyzer:
    """
    Analyzes multi-turn conversations
    Day 4 concepts: tokenization, conversation tracking, stateless calls
    """
    
    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the conversation analyzer
        
        Args:
            model: The OpenAI model to use (default: gpt-4.1-mini)
        """
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.conversations: List[Dict[str, Any]] = []
        self.current_conversation: Optional[Dict[str, Any]] = None
        
        # Pricing for cost calculation (update with current rates)
        self.pricing = {
            "gpt-4.1-mini": {
                "input": 0.15 / 1_000_000,
                "output": 0.60 / 1_000_000
            }
        }
    
    def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """
        Start tracking a new conversation
        
        Args:
            conversation_id: Optional custom ID, otherwise auto-generated
            
        Returns:
            The conversation ID
        """
        if conversation_id is None:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_conversation = {
            "id": conversation_id,
            "messages": [],
            "turns": [],
            "start_time": datetime.now()
        }
        return conversation_id
    
    def add_turn(self, role: str, content: str, response_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a conversation turn
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            response_content: If role is 'user', this is the assistant's response
            
        Returns:
            The turn data
        """
        if self.current_conversation is None:
            self.start_conversation()
        
        # Count tokens for this message
        tokens = self.encoding.encode(content)
        token_count = len(tokens)
        
        turn = {
            "turn_number": len(self.current_conversation["turns"]) + 1,
            "role": role,
            "content": content,
            "token_count": token_count,
            "timestamp": datetime.now()
        }
        
        # If user message, also track assistant response
        if role == "user" and response_content:
            response_tokens = self.encoding.encode(response_content)
            turn["response_content"] = response_content
            turn["response_tokens"] = len(response_tokens)
            turn["total_tokens"] = token_count + len(response_tokens)
        else:
            turn["total_tokens"] = token_count
        
        self.current_conversation["turns"].append(turn)
        self.current_conversation["messages"].append({
            "role": role,
            "content": content
        })
        
        return turn
    
    def analyze_conversation(self, conversation_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Analyze a conversation and return insights
        
        Args:
            conversation_id: ID of conversation to analyze, None for current
            
        Returns:
            Analysis dictionary or None if no conversation found
        """
        if conversation_id:
            # Find specific conversation
            conv = next((c for c in self.conversations if c["id"] == conversation_id), None)
            if not conv:
                return None
        else:
            # Analyze current conversation
            conv = self.current_conversation
        
        if not conv or not conv["turns"]:
            return None
        
        # Calculate statistics
        total_turns = len(conv["turns"])
        total_tokens = sum(turn.get("total_tokens", turn["token_count"]) 
                          for turn in conv["turns"])
        
        # Token breakdown
        user_tokens = sum(turn["token_count"] 
                         for turn in conv["turns"] if turn["role"] == "user")
        assistant_tokens = sum(turn.get("response_tokens", turn["token_count"]) 
                              for turn in conv["turns"] if turn["role"] == "user")
        
        # Average tokens per turn
        avg_tokens_per_turn = total_tokens / total_turns if total_turns > 0 else 0
        
        # Token trend (are messages getting longer?)
        token_trend = []
        for i, turn in enumerate(conv["turns"]):
            if turn["role"] == "user":
                token_trend.append({
                    "turn": i + 1,
                    "tokens": turn.get("total_tokens", turn["token_count"])
                })
        
        # Conversation length
        duration = None
        if conv.get("start_time") and conv["turns"]:
            end_time = conv["turns"][-1]["timestamp"]
            duration = (end_time - conv["start_time"]).total_seconds()
        
        # Build analysis
        analysis = {
            "conversation_id": conv["id"],
            "total_turns": total_turns,
            "total_tokens": total_tokens,
            "user_tokens": user_tokens,
            "assistant_tokens": assistant_tokens,
            "avg_tokens_per_turn": round(avg_tokens_per_turn, 2),
            "token_trend": token_trend,
            "duration_seconds": round(duration, 2) if duration else None,
            "estimated_cost": self._calculate_cost(user_tokens, assistant_tokens)
        }
        
        return analysis
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate estimated cost
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in dollars
        """
        price = self.pricing.get(self.model, {})
        input_cost = input_tokens * price.get("input", 0)
        output_cost = output_tokens * price.get("output", 0)
        return round(input_cost + output_cost, 6)
    
    def generate_insights(self, conversation_id: Optional[str] = None) -> str:
        """
        Generate human-readable insights about the conversation
        
        Args:
            conversation_id: ID of conversation to analyze
            
        Returns:
            Formatted insights string
        """
        analysis = self.analyze_conversation(conversation_id)
        if not analysis:
            return "No conversation data available"
        
        insights = []
        
        # Basic stats
        insights.append(f"📊 Conversation Analysis")
        insights.append(f"Conversation ID: {analysis['conversation_id']}")
        insights.append(f"Total Turns: {analysis['total_turns']}")
        insights.append(f"Total Tokens: {analysis['total_tokens']:,}")
        insights.append(f"Estimated Cost: ${analysis['estimated_cost']:.6f}")
        
        if analysis['duration_seconds']:
            insights.append(f"Duration: {analysis['duration_seconds']:.1f} seconds")
        
        # Token breakdown
        insights.append(f"\n💬 Token Breakdown:")
        insights.append(f"  User tokens: {analysis['user_tokens']:,}")
        insights.append(f"  Assistant tokens: {analysis['assistant_tokens']:,}")
        insights.append(f"  Average per turn: {analysis['avg_tokens_per_turn']:.1f} tokens")
        
        # Token trend analysis
        if len(analysis['token_trend']) > 1:
            insights.append(f"\n📈 Token Trend:")
            first_turn = analysis['token_trend'][0]['tokens']
            last_turn = analysis['token_trend'][-1]['tokens']
            
            if last_turn > first_turn * 1.2:
                change_pct = ((last_turn/first_turn - 1) * 100)
                insights.append(f"  ⬆️ Messages are getting longer (+{change_pct:.1f}%)")
            elif last_turn < first_turn * 0.8:
                change_pct = ((last_turn/first_turn - 1) * 100)
                insights.append(f"  ⬇️ Messages are getting shorter ({change_pct:.1f}%)")
            else:
                insights.append(f"  ➡️ Message length is relatively stable")
        
        # Efficiency insights
        insights.append(f"\n💡 Insights:")
        if analysis['avg_tokens_per_turn'] > 1000:
            insights.append(f"  ⚠️ High token usage per turn - consider optimizing prompts")
        elif analysis['avg_tokens_per_turn'] < 100:
            insights.append(f"  ✅ Efficient token usage")
        
        if analysis['total_tokens'] > 4000:
            insights.append(f"  ⚠️ Conversation exceeds typical context window")
            insights.append(f"  💡 Consider summarizing old messages")
        
        return "\n".join(insights)
    
    def save_conversation(self) -> Optional[str]:
        """
        Save current conversation to history
        
        Returns:
            Conversation ID if saved, None otherwise
        """
        if self.current_conversation and self.current_conversation["turns"]:
            self.current_conversation["end_time"] = datetime.now()
            self.conversations.append(self.current_conversation.copy())
            conv_id = self.current_conversation["id"]
            self.current_conversation = None
            return conv_id
        return None
    
    def get_all_conversations(self) -> List[str]:
        """
        Get list of all conversation IDs
        
        Returns:
            List of conversation IDs
        """
        return [conv["id"] for conv in self.conversations]
    
    def compare_conversations(self, conversation_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Compare multiple conversations
        
        Args:
            conversation_ids: List of conversation IDs to compare
            
        Returns:
            Comparison analysis or None
        """
        comparisons = []
        for conv_id in conversation_ids:
            analysis = self.analyze_conversation(conv_id)
            if analysis:
                comparisons.append(analysis)
        
        if not comparisons:
            return None
        
        # Aggregate statistics
        total_conversations = len(comparisons)
        avg_turns = sum(c["total_turns"] for c in comparisons) / total_conversations
        avg_tokens = sum(c["total_tokens"] for c in comparisons) / total_conversations
        total_cost = sum(c["estimated_cost"] for c in comparisons)
        
        return {
            "total_conversations": total_conversations,
            "avg_turns_per_conversation": round(avg_turns, 2),
            "avg_tokens_per_conversation": round(avg_tokens, 2),
            "total_cost": round(total_cost, 6),
            "conversations": comparisons
        }
    
    def chat_with_analysis(self, user_message: str, system_message: str = "You are a helpful assistant") -> str:
        """
        Chat with OpenAI while tracking the conversation
        Demonstrates Day 4: stateless calls with full history
        
        Args:
            user_message: User's message
            system_message: System prompt
            
        Returns:
            Assistant's response
        """
        if self.current_conversation is None:
            self.start_conversation()
        
        # Add user message
        self.add_turn("user", user_message)
        
        # Prepare messages (Day 4: full conversation history)
        messages = []
        
        # Add system message if not already present
        has_system = any(turn.get("role") == "system" for turn in self.current_conversation["turns"])
        if not has_system:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history
        for turn in self.current_conversation["turns"]:
            if turn["role"] in ["user", "assistant"]:
                content = turn["content"] if turn["role"] == "user" else turn.get("response_content", turn["content"])
                messages.append({
                    "role": turn["role"],
                    "content": content
                })
        
        # Make API call (Day 4: stateless - passing full history)
        openai = OpenAI()
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        assistant_message = response.choices[0].message.content
        
        # Track assistant response
        self.add_turn("assistant", assistant_message)
        
        return assistant_message
    
    def get_token_distribution(self, conversation_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get token distribution across turns
        
        Args:
            conversation_id: ID of conversation, None for current
            
        Returns:
            List of token distributions per turn
        """
        analysis = self.analyze_conversation(conversation_id)
        if not analysis:
            return None
        
        conv = self.current_conversation if not conversation_id else \
               next((c for c in self.conversations if c["id"] == conversation_id), None)
        
        if not conv:
            return None
        
        distribution = []
        for turn in conv["turns"]:
            if turn["role"] == "user":
                distribution.append({
                    "turn": turn["turn_number"],
                    "user_tokens": turn["token_count"],
                    "assistant_tokens": turn.get("response_tokens", 0),
                    "total": turn.get("total_tokens", turn["token_count"])
                })
        
        return distribution
    
    def find_expensive_turns(self, conversation_id: Optional[str] = None, top_n: int = 3) -> Optional[List[Dict[str, Any]]]:
        """
        Find the most expensive conversation turns
        
        Args:
            conversation_id: ID of conversation, None for current
            top_n: Number of top expensive turns to return
            
        Returns:
            List of most expensive turns
        """
        distribution = self.get_token_distribution(conversation_id)
        if not distribution:
            return None
        
        sorted_turns = sorted(distribution, key=lambda x: x["total"], reverse=True)
        return sorted_turns[:top_n]
    
    def print_conversation_flow(self, conversation_id: Optional[str] = None) -> None:
        """
        Print a visual representation of conversation flow
        
        Args:
            conversation_id: ID of conversation, None for current
        """
        conv = self.current_conversation if not conversation_id else \
               next((c for c in self.conversations if c["id"] == conversation_id), None)
        
        if not conv:
            print("No conversation found")
            return
        
        print(f"\n{'='*60}")
        print(f"Conversation: {conv['id']}")
        print(f"{'='*60}\n")
        
        for turn in conv["turns"]:
            role_icon = "👤" if turn["role"] == "user" else "🤖"
            tokens = turn.get("total_tokens", turn["token_count"])
            
            content_preview = turn['content'][:100]
            if len(turn['content']) > 100:
                content_preview += "..."
            
            print(f"{role_icon} Turn {turn['turn_number']} ({tokens} tokens)")
            print(f"   {content_preview}")
            print()
        
        # Summary
        analysis = self.analyze_conversation(conversation_id)
        if analysis:
            print(f"{'='*60}")
            print(f"Total: {analysis['total_turns']} turns, {analysis['total_tokens']:,} tokens")
            print(f"Cost: ${analysis['estimated_cost']:.6f}")
            print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConversationAnalyzer()
    
    # Start a conversation
    conv_id = analyzer.start_conversation("demo_conversation")
    print(f"Started conversation: {conv_id}\n")
    
    # Simulate a multi-turn conversation
    print("Starting conversation...\n")
    
    # Turn 1
    response1 = analyzer.chat_with_analysis("Hi, my name is Kavita Jain and I'm a software engineer")
    print(f"👤 User: Hi, my name is Kavita Jain and I'm a software engineer")
    print(f"🤖 Assistant: {response1}\n")
    
    # Turn 2
    response2 = analyzer.chat_with_analysis("What's my name?")
    print(f"👤 User: What's my name?")
    print(f"🤖 Assistant: {response2}\n")
    
    # Turn 3
    response3 = analyzer.chat_with_analysis("What's my profession?")
    print(f"👤 User: What's my profession?")
    print(f"🤖 Assistant: {response3}\n")
    
    # Print conversation flow
    analyzer.print_conversation_flow()
    
    # Generate insights
    insights = analyzer.generate_insights()
    print(insights)
    
    # Find expensive turns
    expensive = analyzer.find_expensive_turns(top_n=2)
    if expensive:
        print("\n💰 Most expensive turns:")
        for turn in expensive:
            print(f"  Turn {turn['turn']}: {turn['total']} tokens")
    
    # Save conversation
    saved_id = analyzer.save_conversation()
    print(f"\n✅ Conversation saved: {saved_id}")
