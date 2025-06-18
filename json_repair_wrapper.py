"""JSON repair wrapper for fast-graphrag to handle truncated JSON responses from LLMs.

This module patches the format_and_send_prompt function to add JSON repair capabilities,
helping to prevent JSON parsing errors when LLMs return truncated or malformed JSON.

Usage:
    Import this module before using fast-graphrag:

    import json_repair_wrapper
    json_repair_wrapper.patch_fast_graphrag()

    # Then use fast-graphrag normally
    from fast_graphrag import GraphRAG
    # ... rest of your code
"""

import json
import logging
from typing import Any, Tuple, Type

import json_repair
from pydantic import ValidationError

# Set up logging
logger = logging.getLogger(__name__)


def patch_fast_graphrag():
  """Patch fast-graphrag to use JSON repair for handling truncated JSON responses."""
  try:
    # Import the modules we need to patch
    import fast_graphrag._llm._base as llm_base
    from fast_graphrag._models import BaseModelAlias
    from fast_graphrag._prompt import PROMPTS

    # Store the original function
    original_format_and_send_prompt = llm_base.format_and_send_prompt

    # Create the patched version
    async def format_and_send_prompt_with_repair(
      prompt_key: str,
      llm: Any,
      format_kwargs: dict[str, Any],
      response_model: Type[Any],
      **args: Any,
    ) -> Tuple[Any, list[dict[str, str]]]:
      """Patched version of format_and_send_prompt with JSON repair support."""
      try:
        # First try the original function
        return await original_format_and_send_prompt(prompt_key, llm, format_kwargs, response_model, **args)
      except ValidationError as e:
        # If we get a validation error, it might be due to truncated JSON
        logger.warning(f"Validation error in format_and_send_prompt for {prompt_key}, attempting JSON repair: {e}")

        # Try to get raw response without validation
        system_key = prompt_key + "_system"

        if system_key in PROMPTS:
          system = PROMPTS[system_key]
          prompt = PROMPTS[prompt_key + "_prompt"]
          formatted_system = system.format(**format_kwargs)
          formatted_prompt = prompt.format(**format_kwargs)
          raw_response, messages = await llm.send_message(
            system_prompt=formatted_system, prompt=formatted_prompt, response_model=None, **args
          )
        else:
          prompt = PROMPTS[prompt_key]
          formatted_prompt = prompt.format(**format_kwargs)
          raw_response, messages = await llm.send_message(prompt=formatted_prompt, response_model=None, **args)

        # If we got a string response, try to repair it
        if isinstance(raw_response, str):
          try:
            logger.debug(f"Attempting to repair JSON response: {raw_response[:200]}...")
            repaired_json = json_repair.repair_json(raw_response, ensure_ascii=False)
            logger.debug(f"Repaired JSON: {repaired_json[:200]}...")

            # Parse and validate
            parsed_data = json.loads(repaired_json)

            # Handle BaseModelAlias types
            try:
              # Check if it's a BaseModelAlias subclass
              if hasattr(response_model, "Model") and hasattr(response_model, "__bases__"):
                # Try to access the Model attribute
                model_class = response_model.Model
                result = model_class(**parsed_data)
              else:
                # Handle regular Pydantic models
                result = response_model(**parsed_data)

              return result, messages
            except Exception:
              # Fallback: try direct instantiation
              return response_model(**parsed_data), messages

          except Exception as repair_error:
            logger.error(f"Failed to repair JSON: {repair_error}")
            # Re-raise the original validation error
            raise e
        else:
          # If response is not a string, re-raise original error
          raise e

    # Replace the original function with our patched version
    llm_base.format_and_send_prompt = format_and_send_prompt_with_repair

    # Also make it available in the main import
    import fast_graphrag._llm as llm_module

    llm_module.format_and_send_prompt = format_and_send_prompt_with_repair

    logger.info("Successfully patched fast-graphrag with JSON repair support")

  except ImportError as e:
    logger.error(f"Failed to import fast-graphrag modules: {e}")
    raise
  except Exception as e:
    logger.error(f"Failed to patch fast-graphrag: {e}")
    raise


# Additional utility functions


def repair_json_string(json_string: str) -> str:
  """Repair a potentially malformed JSON string.

  Args:
      json_string: The potentially malformed JSON string to repair

  Returns:
      A repaired JSON string
  """
  try:
    # Try to parse as-is first
    json.loads(json_string)
    return json_string
  except json.JSONDecodeError:
    # Use json_repair to fix the JSON
    logger.debug(f"Attempting to repair malformed JSON: {json_string[:200]}...")
    repaired = json_repair.repair_json(json_string, ensure_ascii=False)
    logger.debug(f"Repaired JSON: {repaired[:200]}...")
    return repaired


def test_json_repair():
  """Test the JSON repair functionality with sample truncated JSON."""
  test_cases = [
    # Truncated entities list
    '{"entities": [{"name": "ENTITY1", "type": "TYPE1", "desc": "Description1"}, {"name": "ENTITY2", "type": "TYPE2", "desc": "Description2"',
    # Truncated with missing closing brackets
    '{"entities": [{"name": "TEST", "type": "ORGANIZATION", "desc": "Test org"}], "relationships": [{"source": "TEST", "target": "OTHER", "desc": "Relates to"',
    # Empty response
    "",
  ]

  for i, test_json in enumerate(test_cases):
    print(f"\nTest case {i+1}:")
    print(f"Original: {test_json}")
    try:
      repaired = repair_json_string(test_json)
      print(f"Repaired: {repaired}")
      parsed = json.loads(repaired)
      print(f"Parsed successfully: {type(parsed)}")
    except Exception as e:
      print(f"Failed to repair: {e}")


if __name__ == "__main__":
  # Run tests
  test_json_repair()
