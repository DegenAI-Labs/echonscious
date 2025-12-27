import json
import os
from typing import Optional, Tuple, Dict, Any

from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic

from llm_local import get_vllm_client


# Initialize clients with API keys from environment variables
# Set these in your .env file or export them:
# export OPENAI_API_KEY="sk-..."
# export ANTHROPIC_API_KEY="sk-ant-..."
# export AZURE_OPENAI_ENDPOINT="https://..."
# export AZURE_OPENAI_API_KEY="..."

def get_openai_client():
    """Get standard OpenAI client"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def get_azure_client():
    """Get Azure OpenAI client (simplified, no managed identity)"""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not endpoint or not api_key:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables required"
        )
    
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def get_anthropic_client():
    """Get Anthropic client"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return Anthropic(api_key=api_key)


def get_response_from_gpt(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Tuple[Dict[str, Any], None, None]:
    """
    Standard OpenAI API call (non-Azure)
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4-turbo")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (json_dict, None, None) matching the expected signature
    """
    client = get_openai_client()
    
    print(f"*** model = {model}")
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        response_format={"type": "json_object"},
    )
    
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    
    return json_dict, None, None


def get_response_from_gpt_azure(
    model: str,
    messages: list,
    reasoning_effort: str = "low",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Tuple[Dict[str, Any], None, None]:
    """
    Azure OpenAI API call
    
    Args:
        model: Model deployment name (e.g., "gpt-4o", "o3-mini")
        messages: List of message dicts
        reasoning_effort: For o3-mini models
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        
    Returns:
        Tuple of (json_dict, None, None)
    """
    client = get_azure_client()
    
    print(f"*** model = {model}")
    
    if model == "o3-mini":
        # For o3-mini, we use reasoning_effort parameter
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_tokens,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            response_format={"type": "json_object"},
        )
    
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    
    return json_dict, None, None


def get_response_from_anthropic(
    model: str,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    system_prompt: Optional[str] = None,
) -> Tuple[Dict[str, Any], None, Optional[str]]:
    """
    Anthropic/Claude API call
    
    Args:
        model: Model name (e.g., "claude-sonnet-4-20250514")
        messages: List of message dicts
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: Optional system prompt (Anthropic uses separate parameter)
        
    Returns:
        Tuple of (json_dict, None, thought) where thought may contain extended thinking
    """
    client = get_anthropic_client()
    
    print(f"*** model = {model}")
    
    # Anthropic API uses separate system parameter
    # Extract system message if present in messages
    anthropic_messages = []
    extracted_system = system_prompt
    
    for msg in messages:
        if msg["role"] == "system":
            extracted_system = msg["content"]
        else:
            anthropic_messages.append(msg)
    
    # Make the API call
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=extracted_system or "",
        messages=anthropic_messages,
    )
    
    # Extract content
    content = response.content[0].text
    
    # Check for extended thinking (if model used thinking)
    thought = None
    if hasattr(response.content[0], 'thinking'):
        thought = response.content[0].thinking
    
    # Parse as JSON
    try:
        json_dict = json.loads(content)
    except json.JSONDecodeError:
        # If response isn't valid JSON, wrap it
        json_dict = {"content": content}
    
    return json_dict, None, thought


def my_process_choice(choice):
    """Legacy function - preserved for compatibility"""
    return choice


response_processor_factory = None


def extract_thinking_and_content(raw_content: str) -> Tuple[Optional[str], str]:
    """Extract thinking content and main response from raw content.

    Args:
        raw_content: The raw response content that may contain <think>...</think> tags

    Returns:
        tuple: (thought, cleaned_content) where thought is the extracted thinking
               and cleaned_content is the main response after the thinking tags
    """
    start_tag = "<think>"
    end_tag = "</think>"

    start_index = raw_content.find(start_tag)
    end_index = raw_content.find(end_tag, start_index + len(start_tag))

    if start_index != -1 and end_index != -1:
        # Extract the thought content between the tags
        thought = raw_content[start_index + len(start_tag) : end_index].strip()
        # The main content comes after the closing tag
        content = raw_content[end_index + len(end_tag) :].strip()
        return thought, content
    else:
        # No thinking tags found, return None for thought and full content
        return None, raw_content.strip()


def get_response_from_local(
    model: str,
    messages: list,
    is_thinking: bool = False,
) -> Tuple[Dict[str, Any], None, Optional[str]]:
    """
    Use the local vLLM client for making requests
    
    Args:
        model: Model name for vLLM
        messages: List of message dicts
        is_thinking: Whether to extract <think> tags
        
    Returns:
        Tuple of (content_dict, None, thought)
    """
    vllm_client = get_vllm_client(model_name=model)

    # Check if server is available
    if not vllm_client.is_available():
        raise Exception(
            f"vLLM server is not available. Start it with: python -m vllm.entrypoints.openai.api_server --model {model} --port 8000"
        )

    # Prepare the request
    response = vllm_client.send_request(
        messages=messages, temperature=0.0, max_tokens=800
    )

    # Ensure response is parsed as JSON
    if isinstance(response, str):
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from vLLM backend.")

    content = response["choices"][0]["message"]["content"].strip()

    print("content = ", content)

    if is_thinking:
        thought, content = extract_thinking_and_content(content)
    else:
        thought = None

    content_dct = content
    if isinstance(content_dct, str):
        try:
            content_dct = json.loads(content_dct)
        except json.JSONDecodeError:
            content_dct = {"content": content}

    return content_dct, None, thought


def main():
    """Test the different API backends"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Respond in JSON format with a 'answer' key."},
    ]

    # Test standard OpenAI
    print("\n" + "="*60)
    print("Testing standard OpenAI API...")
    print("="*60)
    try:
        result, _, _ = get_response_from_gpt(
            model="gpt-4o-mini", 
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        print(f"✅ OpenAI Result: {result}")
    except Exception as e:
        print(f"❌ OpenAI error: {e}")
        print("💡 Set OPENAI_API_KEY environment variable")

    # Test Anthropic
    print("\n" + "="*60)
    print("Testing Anthropic API...")
    print("="*60)
    try:
        result, _, thought = get_response_from_anthropic(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "What is the capital of France? Respond in JSON format with an 'answer' key."}],
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            max_tokens=200
        )
        print(f"✅ Anthropic Result: {result}")
        if thought:
            print(f"💭 Thought: {thought}")
    except Exception as e:
        print(f"❌ Anthropic error: {e}")
        print("💡 Set ANTHROPIC_API_KEY environment variable")

    # Test local vLLM
    print("\n" + "="*60)
    print("Testing local vLLM backend...")
    print("="*60)
    try:
        result, _, thought = get_response_from_local(
            model="Qwen/Qwen2.5-7B-Instruct", 
            messages=messages, 
            is_thinking=False
        )
        print(f"✅ Local vLLM Result: {result}")
        if thought:
            print(f"💭 Thought: {thought}")
    except Exception as e:
        print(f"❌ Local vLLM error: {e}")
        print("💡 To use local backend, start vLLM server:")
        print("   python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-32B-Instruct --port 8000")


if __name__ == "__main__":
    main()

# Usage:
# export OPENAI_API_KEY="sk-..."
# export ANTHROPIC_API_KEY="sk-ant-..."
# PYTHONPATH=. python api_call.py
