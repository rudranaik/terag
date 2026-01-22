"""
Multi-LLM Configuration Example

Demonstrates how to configure TERAG to use different LLM providers (Groq, OpenAI)
for higher quality entity extraction.
"""

import os
from terag import TERAG, TERAGConfig

def main():
    # Mock data
    chunks = [{"content": "SpaceX successfully launched Starship from Boca Chica."}]

    print("--- Configuration for Groq (Default) ---")
    config_groq = TERAGConfig(
        use_llm_for_ner=True,
        llm_provider="groq",
        # llm_api_key="gsk-..." # Optional: set here or use env var GROQ_API_KEY
        model="openai/gpt-oss-20b"
    )
    print(f"Configured for Groq: {config_groq.llm_provider}")
    
    
    print("\n--- Configuration for OpenAI ---")
    config_openai = TERAGConfig(
        use_llm_for_ner=True,
        llm_provider="openai",
        # llm_api_key="sk-..." # Optional: set here or use env var OPENAI_API_KEY
        model="gpt-3.5-turbo"
    )
    print(f"Configured for OpenAI: {config_openai.llm_provider}")

    print("\nTo run with actual API calls:")
    print("1. Set GROQ_API_KEY or OPENAI_API_KEY environment variables")
    print("2. Run: python examples/multi_llm_example.py")
    
    # Example construction (will fallback to regex if no keys present)
    print("\nAttempting build with Groq config...")
    terag = TERAG.from_chunks(chunks, config=config_groq, verbose=True)

if __name__ == "__main__":
    main()
