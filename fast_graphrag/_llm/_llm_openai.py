"""OpenAI-compatible LLM Services module."""

import asyncio
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, List, Literal, Optional, Tuple, Type, cast

import instructor
import json_repair
import numpy as np
import tiktoken
from aiolimiter import AsyncLimiter
from openai import APIConnectionError, AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from fast_graphrag._exceptions import LLMServiceNoResponseError
from fast_graphrag._types import BaseModelAlias
from fast_graphrag._utils import logger

from ._base import TOKEN_PATTERN, BaseEmbeddingService, BaseLLMService, NoopAsyncContextManager, T_model

TIMEOUT_SECONDS = 180.0


@dataclass
class OpenAILLMService(BaseLLMService):
  """LLM Service for OpenAI LLMs."""

  model: str = field(default="gpt-4o-mini")
  mode: instructor.Mode = field(default=instructor.Mode.JSON)
  client: Literal["openai", "azure"] = field(default="openai")
  api_version: Optional[str] = field(default=None)
  use_json_repair: bool = field(default=True)  # Enable JSON repair by default

  def __post_init__(self):
    self.encoding = None
    try:
      self.encoding = tiktoken.encoding_for_model(self.model)
    except Exception as e:
      logger.info(f"LLM: failed to load tokenizer for model '{self.model}' ({e}). Falling back to naive tokenization.")
      self.encoding = None

    self.llm_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    self.llm_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    self.llm_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )

    if self.client == "azure":
      assert (
        self.base_url is not None and self.api_version is not None
      ), "Azure OpenAI requires a base url and an api version."
      client = AsyncAzureOpenAI(
        azure_endpoint=self.base_url,
        api_key=self.api_key,
        api_version=self.api_version,
        timeout=TIMEOUT_SECONDS,
      )
    elif self.client == "openai":
      client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=TIMEOUT_SECONDS)
    else:
      raise ValueError("Invalid client type. Must be 'openai' or 'azure'")

    self.llm_async_client = instructor.from_openai(client, mode=self.mode)
    self._raw_client = client  # Keep reference to raw client for JSON repair fallback

    logger.debug("Initialized OpenAILLMService with patched OpenAI client.")

  def count_tokens(self, text: str) -> int:
    """Returns the number of tokens for a given text using the encoding appropriate for the model."""
    if self.encoding:
      return len(self.encoding.encode(text))
    else:
      return len(TOKEN_PATTERN.findall(text))

  async def send_message(
    self,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    response_model: Type[T_model] | None = None,
    **kwargs: Any,
  ) -> Tuple[T_model, list[dict[str, str]]]:
    """Send a message to the language model and receive a response.

    Args:
        prompt (str): The input message to send to the language model.
        model (str): The name of the model to use. Defaults to the model provided in the config.
        system_prompt (str, optional): The system prompt to set the context for the conversation. Defaults to None.
        history_messages (list, optional): A list of previous messages in the conversation. Defaults to empty.
        response_model (Type[T], optional): The Pydantic model to parse the response. Defaults to None.
        **kwargs: Additional keyword arguments that may be required by specific LLM implementations.

    Returns:
        str: The response from the language model.
    """
    async with self.llm_max_requests_concurrent:
      async with self.llm_per_minute_limiter:
        async with self.llm_per_second_limiter:
          try:
            logger.debug(f"Sending message with prompt: {prompt}")
            model = self.model
            messages: list[dict[str, str]] = []

            if system_prompt:
              messages.append({"role": "system", "content": system_prompt})
              logger.debug(f"Added system prompt: {system_prompt}")

            if history_messages:
              messages.extend(history_messages)
              logger.debug(f"Added history messages: {history_messages}")

            messages.append({"role": "user", "content": prompt})

            try:
              # First attempt with instructor
              llm_response: T_model = await self.llm_async_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore
                response_model=response_model.Model
                if response_model and issubclass(response_model, BaseModelAlias)
                else response_model,
                **kwargs,
                max_retries=AsyncRetrying(
                  stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
                ),
              )
            except ValidationError as e:
              # If we get a validation error and JSON repair is enabled, try to repair
              if self.use_json_repair and response_model:
                logger.warning(f"Validation error, attempting JSON repair: {e}")
                llm_response = await self._send_with_json_repair(
                  model=model, messages=messages, response_model=response_model, **kwargs
                )
              else:
                raise

            if not llm_response:
              logger.error("No response received from the language model.")
              raise LLMServiceNoResponseError("No response received from the language model.")

            messages.append(
              {
                "role": "assistant",
                "content": llm_response.model_dump_json() if isinstance(llm_response, BaseModel) else str(llm_response),
              }
            )
            logger.debug(f"Received response: {llm_response}")

            if response_model and issubclass(response_model, BaseModelAlias):
              llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

            return llm_response, messages
          except Exception:
            logger.exception("An error occurred:", exc_info=True)
            raise

  async def _send_with_json_repair(
    self,
    model: str,
    messages: list[dict[str, str]],
    response_model: Type[T_model],
    **kwargs: Any,
  ) -> T_model:
    """Send message with JSON repair fallback."""
    try:
      # Get raw response without instructor parsing
      raw_response = await self._raw_client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        response_format={"type": "json_object"} if self.mode == instructor.Mode.JSON else None,
        **kwargs,
      )

      if raw_response.choices and raw_response.choices[0].message.content:
        raw_json = raw_response.choices[0].message.content
        logger.debug(f"Raw JSON response: {raw_json[:200]}...")

        # Repair the JSON
        repaired_json = json_repair.repair_json(raw_json, ensure_ascii=False)
        logger.debug(f"Repaired JSON: {repaired_json[:200]}...")

        # Parse and validate
        parsed_data = json.loads(repaired_json)

        # Handle BaseModelAlias types
        if issubclass(response_model, BaseModelAlias):
          return response_model.Model(**parsed_data)
        else:
          return response_model(**parsed_data)
      else:
        raise LLMServiceNoResponseError("No content in response")
    except Exception as e:
      logger.error(f"Failed to repair JSON: {e}")
      raise


@dataclass
class OpenAIEmbeddingService(BaseEmbeddingService):
  """Base class for Language Model implementations."""

  embedding_dim: int = field(default=1536)
  max_elements_per_request: int = field(default=32)
  model: Optional[str] = field(default="text-embedding-3-small")
  client: Literal["openai", "azure"] = field(default="openai")
  api_version: Optional[str] = field(default=None)

  def __post_init__(self):
    self.embedding_max_requests_concurrent = (
      asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
    )
    self.embedding_per_minute_limiter = (
      AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
    )
    self.embedding_per_second_limiter = (
      AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
    )
    if self.client == "azure":
      assert (
        self.base_url is not None and self.api_version is not None
      ), "Azure OpenAI requires a base url and an api version."
      self.embedding_async_client = AsyncAzureOpenAI(
        azure_endpoint=self.base_url, api_key=self.api_key, api_version=self.api_version
      )
    elif self.client == "openai":
      self.embedding_async_client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
    else:
      raise ValueError("Invalid client type. Must be 'openai' or 'azure'")
    logger.debug("Initialized OpenAIEmbeddingService with OpenAI client.")

  async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    try:
      """Get the embedding representation of the input text.

            Args:
                texts (str): The input text to embed.
                model (str, optional): The name of the model to use. Defaults to the model provided in the config.

            Returns:
                list[float]: The embedding vector as a list of floats.
            """
      logger.debug(f"Getting embedding for texts: {texts}")
      model = model or self.model
      if model is None:
        raise ValueError("Model name must be provided.")

      batched_texts = [
        texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
        for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
      ]
      response = await asyncio.gather(*[self._embedding_request(b, model) for b in batched_texts])

      data = chain(*[r.data for r in response])
      embeddings = np.array([dp.embedding for dp in data])
      logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

      return embeddings
    except Exception:
      logger.exception("An error occurred:", exc_info=True)
      raise

  @retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, TimeoutError)),
  )
  async def _embedding_request(self, input: List[str], model: str) -> Any:
    async with self.embedding_max_requests_concurrent:
      async with self.embedding_per_minute_limiter:
        async with self.embedding_per_second_limiter:
          return await self.embedding_async_client.embeddings.create(
            model=model, input=input, dimensions=self.embedding_dim, encoding_format="float"
          )
