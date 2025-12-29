"""
Dataset format converters for quantization calibration.

This module provides a namespace class `DatasetFmt` with static methods for converting
various dataset formats into the chat completion format expected by llm-compressor.
"""

from typing import Optional, Dict, Any, List

class DatasetFmt:
    """
    Namespace class for dataset format converters.
    
    All methods are static - do not instantiate this class.
    """
    
    @staticmethod
    def sharegpt(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert ShareGPT conversation format to chat completion format.
        
        ShareGPT format uses "from" and "value" keys with roles: system, human, gpt.
        
        Args:
            data: Single entry from ShareGPT dataset with "conversations" field
            
        Returns:
            List of message dicts with "role" and "content" keys
        """
        role_mapping = {
            "system": "system",
            "human": "user",
            "gpt": "assistant"
        }
        
        if "conversations" not in data:
            raise ValueError("ShareGPT format requires 'conversations' field")
        
        messages = []
        for entry in data["conversations"]:
            role = role_mapping.get(entry.get("from"), "user")
            messages.append({
                "role": role,
                "content": entry.get("value", "")
            })
        
        return messages
    
    @staticmethod
    def alpaca(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert Alpaca instruction format to chat completion format.
        
        Alpaca format uses: instruction, input, output fields.
        
        Args:
            data: Single entry from Alpaca dataset
            
        Returns:
            List of message dicts with "role" and "content" keys
        """
        instruction = data.get("instruction", "")
        input_text = data.get("input", "")
        output = data.get("output", "")
        
        messages = []
        
        # Build user message
        if input_text:
            user_content = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_content = instruction
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Add assistant response if present
        if output:
            messages.append({
                "role": "assistant",
                "content": output
            })
        
        return messages
    
    @staticmethod
    def raw_text(data: Dict[str, Any], instruction: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Convert raw text to chat completion format with instruction prompt.
        
        Args:
            data: Single entry with "text" field containing raw content
            instruction: Optional instruction to prepend. If None, uses "Continue"
            
        Returns:
            List with single user message containing instruction and text
        """
        if "text" not in data:
            raise ValueError("raw_text format requires 'text' field")
        
        if instruction is None:
            instruction = "Continue"
        
        messages = [{
            "role": "user",
            "content": f"{instruction}:\n\n{data['text']}"
        }]
        
        return messages
    
    @staticmethod
    def chat_completion(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Pass through already-formatted chat completion data.
        
        Args:
            data: Single entry that should already be in messages format
            
        Returns:
            The messages list directly
        """
        if "messages" not in data:
            raise ValueError("chat_completion format requires 'messages' field")
        
        return data["messages"]
    
    @staticmethod
    def prompt_answer(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Convert prompt/answer format to chat completion format.
        
        Args:
            data: Single entry with "prompt" and "answer" fields
            
        Returns:
            List of user/assistant message dicts
        """
        messages = [
            {
                "role": "user",
                "content": data.get("prompt", "")
            }
        ]
        
        if "answer" in data and data["answer"]:
            messages.append({
                "role": "assistant",
                "content": data["answer"]
            })
        
        return messages
