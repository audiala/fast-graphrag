# Using JSON Repair with fast-graphrag

This guide explains how to use the `json_repair` library to handle truncated or malformed JSON responses from LLMs when using fast-graphrag.

## Problem

When using fast-graphrag with LLMs, you may encounter JSON parsing errors like:

```
1 validation error for Graph
  Invalid JSON: EOF while parsing a list at line 204 column 5
```

These errors occur when the LLM returns truncated JSON responses, which can happen due to token limits or other issues.

## Solution

### Option 1: Using the JSON Repair Wrapper (Recommended)

1. First, ensure you have `json-repair` installed:
   ```bash
   pip install json-repair
   ```

2. Use the provided `json_repair_wrapper.py` file by importing it before using fast-graphrag:

   ```python
   # Import and apply the JSON repair patch
   import json_repair_wrapper
   json_repair_wrapper.patch_fast_graphrag()

   # Now import and use fast-graphrag normally
   from fast_graphrag import GraphRAG
   from fast_graphrag._llm import OpenAILLMService

   # Your code continues as normal
   llm = OpenAILLMService(model="gpt-4.1-mini")
   graphrag = GraphRAG(
       config=GraphRAG.Config(
           llm_service=llm,
           # ... other config
       )
   )
   ```

### Option 2: Custom LLM Service with JSON Repair

Create a custom LLM service that inherits from OpenAILLMService and adds JSON repair:

```python
import json
import json_repair
from typing import Any, Tuple, Type, Optional
from pydantic import BaseModel, ValidationError
from fast_graphrag._llm import OpenAILLMService
from fast_graphrag._utils import logger

class JSONRepairLLMService(OpenAILLMService):
    """OpenAI LLM Service with JSON repair for handling truncated responses."""
    
    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[Any] | None = None,
        **kwargs: Any,
    ) -> Tuple[Any, list[dict[str, str]]]:
        """Send message with JSON repair fallback."""
        try:
            # Try the normal approach first
            return await super().send_message(
                prompt, system_prompt, history_messages, response_model, **kwargs
            )
        except ValidationError as e:
            if response_model is None:
                raise
                
            logger.warning(f"JSON validation error, attempting repair: {e}")
            
            # Get raw response without validation
            raw_result, messages = await super().send_message(
                prompt, system_prompt, history_messages, None, **kwargs
            )
            
            if isinstance(raw_result, str):
                # Repair the JSON
                repaired_json = json_repair.repair_json(raw_result, ensure_ascii=False)
                parsed_data = json.loads(repaired_json)
                
                # Validate with the response model
                if hasattr(response_model, 'Model'):
                    # Handle BaseModelAlias types
                    return response_model.Model(**parsed_data), messages
                else:
                    # Handle regular Pydantic models
                    return response_model(**parsed_data), messages
            else:
                raise

# Use the custom service
llm = JSONRepairLLMService(
    model="gpt-4.1-mini",
    base_url="https://your-endpoint.openai.azure.com/",
    api_key="your-api-key"
)
```

### Option 3: Manual JSON Repair

If you encounter JSON errors during processing, you can manually repair the JSON:

```python
import json_repair

def safe_parse_json(json_string: str) -> dict:
    """Safely parse JSON with automatic repair."""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        repaired = json_repair.repair_json(json_string, ensure_ascii=False)
        return json.loads(repaired)
```

## How JSON Repair Works

The `json_repair` library can fix common JSON issues including:

- Missing closing brackets `]` or `}`
- Missing quotes around keys
- Truncated strings
- Missing commas
- Trailing commas
- Single quotes instead of double quotes
- Unescaped characters

For truncated JSON responses from LLMs, it intelligently adds the missing closing brackets and quotes to create valid JSON.

## Testing

You can test the JSON repair functionality:

```python
python json_repair_wrapper.py
```

This will run test cases with various types of truncated JSON to verify the repair functionality.

## Troubleshooting

1. **Import Order**: Make sure to import and patch before importing fast-graphrag
2. **Logging**: Enable debug logging to see repair attempts:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
3. **Fallback**: If JSON repair fails, the original error will be raised

## Performance Considerations

- JSON repair adds a small overhead only when JSON parsing fails
- The first attempt always uses the standard parsing
- Repair is only triggered for `ValidationError` exceptions
- For best performance, ensure your LLM prompts encourage complete JSON output 