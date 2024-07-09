Welcome to AIOS. Model: claude-3-5-sonnet-20240620

[1mπ [0m```sh
ls -l examples tiny_ai_client

````
[31mPress enter to execute, or 'N' to cancel.[0m
[F[K[F[K[2mexamples:
total 40
drwxr-xr-x  4 user  role   128  4 Jun 00:29 __pycache__
-rw-r--r--  1 user  role  2271  7 Jul 02:18 anthropic_.py
-rw-r--r--  1 user  role  1593  9 Jul 11:28 gemini_.py
-rw-r--r--  1 user  role  2195  7 Jul 15:04 groq_.py
-rw-r--r--  1 user  role   992  7 Jul 14:39 ollama_.py
-rw-r--r--  1 user  role  2349  7 Jul 02:50 openai_.py

tiny_ai_client:
total 128
-rw-r--r--   1 user  role    32  4 Jun 01:13 __init__.py
drwxr-xr-x  22 user  role   704  9 Jul 11:28 __pycache__
-rw-r--r--   1 user  role  7779  7 Jul 02:17 anthropic_.py
-rw-r--r--   1 user  role  4348  9 Jul 11:28 gemini_.py
-rw-r--r--   1 user  role  6366  7 Jul 15:03 groq_.py
-rw-r--r--   1 user  role  9592  7 Jul 03:00 models.py
-rw-r--r--   1 user  role  2844 13 Jun 11:19 ollama_.py
-rw-r--r--   1 user  role  6863  7 Jul 02:49 openai_.py
-rw-r--r--   1 user  role  1111  5 Jun 13:30 tools.py
-rw-r--r--   1 user  role  1656 14 Jun 18:10 transformers_.py
-rw-r--r--   1 user  role   258  4 Jun 10:36 vision.py[0m
[1mπ [0m```sh
cat tiny_ai_client/anthropic_.py tiny_ai_client/gemini_.py
````

[31mPress enter to execute, or 'N' to cancel.[0m
[F[K[F[K[2mimport os
from typing import Any, Callable, Dict, List, Union, Generator, AsyncGenerator

from anthropic import Anthropic, AsyncAnthropic

from tiny_ai_client.models import LLMClientWrapper, Message, ToolCall
from tiny_ai_client.tools import function_to_json
from tiny_ai_client.vision import encode_pil_image

class AnthropicClientWrapper(LLMClientWrapper):
def **init**(self, model_name: str, tools: List[Union[Callable, Dict]]):
self.model_name = model_name
self.client = Anthropic(
api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
self.async_client = AsyncAnthropic(
api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
self.tools = tools
self.tools_json = [
function_to_json(tool, "input_schema")["function"] for tool in tools
]

    def build_model_input(self, messages: List["Message"]) -> Any:
        input_messages = []
        system = None
        for message in messages:
            if message.role == "system":
                system = message.text
                continue
            elif message.tool_call:
                content_in = [
                    {
                        "type": "tool_use",
                        "name": message.tool_call.name,
                        "id": message.tool_call.id_,
                        "input": message.tool_call.parameters,
                    }
                ]
                content_out = [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call.id_,
                        "content": [
                            {"type": "text", "text": str(message.tool_call.result)}
                        ],
                    }
                ]
                input_messages.append(
                    {
                        "role": "assistant",
                        "content": content_in,
                    }
                )
                input_messages.append(
                    {
                        "role": "user",
                        "content": content_out,
                    }
                )
                input_messages.append(
                    {
                        "role": "assistant",
                        "content": "Acknowledged.",
                    }
                )
                continue
            else:
                content = []
                role = message.role
                if message.images:
                    for image in message.images:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_pil_image(image),
                                },
                            }
                        )
                if message.text is not None:
                    content.append(
                        {
                            "type": "text",
                            "text": message.text,
                        }
                    )
                model_input_message = {
                    "role": role,
                    "content": content,
                }
            input_messages.append(model_input_message)
        return input_messages, system

    def call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        kwargs = {}
        input_messages, system = model_input
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json
        response = self.client.with_options(timeout=timeout).messages.create(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        )
        response_content = response.content[0]
        if response_content.type == "text":
            return Message(text=response_content.text, role="assistant")

        if response_content.type == "tool_use":
            print(response_content)
            return Message(
                role="assistant",
                text=None,
                tool_call=ToolCall(
                    id_=response_content.id,
                    parameters=response_content.input,
                    name=response_content.name,
                ),
            )

    def stream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> Generator[str, None, None]:
        kwargs = {}
        input_messages, system = self.build_model_input(chat)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json

        with self.client.with_options(timeout=timeout).messages.stream(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        ) as stream:
            for text in stream.text_stream:
                yield text

    async def async_call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        kwargs = {}
        input_messages, system = model_input
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json
        response = await self.async_client.with_options(
            timeout=timeout
        ).messages.create(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        )
        response_content = response.content[0]
        if response_content.type == "text":
            return Message(text=response_content.text, role="assistant")

        if response_content.type == "tool_use":
            return Message(
                role="assistant",
                text=None,
                tool_call=ToolCall(
                    id_=response_content.id,
                    parameters=response_content.input,
                    name=response_content.name,
                ),
            )

    async def astream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> AsyncGenerator[str, None]:
        kwargs = {}
        input_messages, system = self.build_model_input(chat)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            kwargs["max_tokens"] = max_new_tokens
        if system is not None:
            kwargs["system"] = system
        if self.tools_json:
            kwargs["tools"] = self.tools_json

        async with self.async_client.with_options(timeout=timeout).messages.stream(
            model=self.model_name,
            messages=input_messages,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import google.generativeai as genai

from tiny_ai_client.models import LLMClientWrapper, Message
from tiny_ai_client.tools import function_to_json

class GeminiClientWrapper(LLMClientWrapper):
def **init**(self, model_name: str, tools: List[Union[Callable, Dict]]):
self.model_name = model_name
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
self.tools = tools
self.tools_json = [function_to_json(tool) for tool in tools]
if len(self.tools_json) > 0:
raise ValueError("Gemini does not support tools") # self.tools_obj = [ # genai.types.FunctionDeclaration( # name=tool["function"]["name"], # description=tool["function"]["description"], # parameters=genai.types.content_types.strip_titles( # tool["function"]["parameters"] # ), # ) # for tool in self.tools_json # ]

    def build_model_input(self, messages: List["Message"]) -> Any:
        history = []
        local_messages = deepcopy(messages)
        system = None
        message = None

        for message in local_messages:
            if message.role == "system":
                system = message.text
                continue
            else:
                if message.role not in ["user", "assistant"]:
                    raise ValueError(f"Invalid role for Gemini: {message.role}")
                role = "user" if message.role == "user" else "model"
                parts = []
                if message.text is not None:
                    parts.append(message.text)
                if message.images is not None:
                    parts.extend(message.images)
                history.append(
                    {
                        "role": role,
                        "parts": parts,
                    }
                )

        return (system, history)

    def call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        system, history = model_input

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = model.generate_content(history)
        response = response.candidates[0].content.parts[0]
        if response.function_call.name != "":
            raise ValueError("Function calls are not supported in Gemini")
        elif response.text is not None:
            return Message(role="assistant", text=response.text)
        raise ValueError("Invalid response from Gemini")

    async def async_call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        system, history = model_input

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = await model.generate_content_async(history)
        response = response.candidates[0].content.parts[0]
        if response.function_call.name != "":
            raise ValueError("Function calls are not supported in Gemini")
        elif response.text is not None:
            return Message(role="assistant", text=response.text)
        raise ValueError("Invalid response from Gemini")[0m

[1mπ [0m```sh
cat > tiny*ai_client/gemini*.py << EOL
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union, Generator, AsyncGenerator

import google.generativeai as genai

from tiny_ai_client.models import LLMClientWrapper, Message
from tiny_ai_client.tools import function_to_json

class GeminiClientWrapper(LLMClientWrapper):
def **init**(self, model_name: str, tools: List[Union[Callable, Dict]]):
self.model_name = model_name
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
self.tools = tools
self.tools_json = [function_to_json(tool) for tool in tools]
if len(self.tools_json) > 0:
raise ValueError("Gemini does not support tools")

    def build_model_input(self, messages: List["Message"]) -> Any:
        history = []
        local_messages = deepcopy(messages)
        system = None
        message = None

        for message in local_messages:
            if message.role == "system":
                system = message.text
                continue
            else:
                if message.role not in ["user", "assistant"]:
                    raise ValueError(f"Invalid role for Gemini: {message.role}")
                role = "user" if message.role == "user" else "model"
                parts = []
                if message.text is not None:
                    parts.append(message.text)
                if message.images is not None:
                    parts.extend(message.images)
                history.append(
                    {
                        "role": role,
                        "parts": parts,
                    }
                )

        return (system, history)

    def call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        system, history = model_input

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = model.generate_content(history)
        response = response.candidates[0].content.parts[0]
        if response.function_call.name != "":
            raise ValueError("Function calls are not supported in Gemini")
        elif response.text is not None:
            return Message(role="assistant", text=response.text)
        raise ValueError("Invalid response from Gemini")

    def stream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> Generator[str, None, None]:
        system, history = self.build_model_input(chat)

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = model.generate_content(history, stream=True)
        for chunk in response:
            yield chunk.text

    async def async_call_llm_provider(
        self,
        model_input: Any,
        temperature: int | None,
        max_new_tokens: int | None,
        timeout: int,
    ) -> str:
        system, history = model_input

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = await model.generate_content_async(history)
        response = response.candidates[0].content.parts[0]
        if response.function_call.name != "":
            raise ValueError("Function calls are not supported in Gemini")
        elif response.text is not None:
            return Message(role="assistant", text=response.text)
        raise ValueError("Invalid response from Gemini")

    async def astream(
        self,
        max_new_tokens: int | None,
        temperature: int | None,
        timeout: int,
        chat: List["Message"],
    ) -> AsyncGenerator[str, None]:
        system, history = self.build_model_input(chat)

        generation_config_kwargs = {}
        if temperature is not None:
            generation_config_kwargs["temperature"] = temperature
        if max_new_tokens is not None:
            generation_config_kwargs["max_output_tokens"] = max_new_tokens

        generation_config = genai.GenerationConfig(**generation_config_kwargs)
        model = genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config=generation_config,
        )

        model.start_chat(history=history)
        response = await model.generate_content_async(history, stream=True)
        async for chunk in response:
            yield chunk.text

EOL

````
[31mPress enter to execute, or 'N' to cancel.[0m
[F[K[F[K[2m[0m
[1mπ [0m```sh
cat examples/anthropic_.py examples/gemini_.py
````

[31mPress enter to execute, or 'N' to cancel.[0m
[F[K[F[K[2mimport asyncio

from PIL import Image as PIL_Image
from pydantic import BaseModel, Field

from tiny_ai_client import AI, AsyncAI

class WeatherParams(BaseModel):
location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
unit: str = Field(
"celsius", description="Temperature unit", enum=["celsius", "fahrenheit"]
)

def get_current_weather(weather: WeatherParams):
"""
Get the current weather in a given location
"""
return {
"abc": f"Getting the current weather in {weather.location} in {weather.unit}."
}

def get_images():
return [PIL_Image.open("assets/kirk.jpg"), PIL_Image.open("assets/spock.jpg")]

async def async_ai_main():
print("### ASYNC AI ###")
ai = AsyncAI(
model_name="claude-3-sonnet-20240229",
system="You are a helpful assistant capable of using tools.",
max_new_tokens=1024,
tools=[get_current_weather],
)
response = await ai("Please get the current weather in celsius for San Francisco.")
print(f"{response=}")
response = await ai("Did it work?")
print(f"{response=}")
response = await ai("Who is on the images?", images=get_images())
print(f"{response=}")
print(f"{ai.chat=}")

    print("\n### ASYNC AI STREAMING ###")
    async for chunk in ai.astream("Tell me a short story about a brave astronaut."):
        print(chunk, end="", flush=True)
    print("\n")

def main():
print("### SYNC AI ###")
ai = AI(
model_name="claude-3-sonnet-20240229",
system="You are a helpful assistant capable of using tools.",
max_new_tokens=1024,
tools=[get_current_weather],
)
response = ai("Please get the current weather in celsius for San Francisco.")
print(f"{response=}")
response = ai("Did it work?")
print(f"{response=}")
response = ai("Who is on the images?", images=get_images())
print(f"{response=}")
print(f"{ai.chat=}")

    print("\n### SYNC AI STREAMING ###")
    for chunk in ai.stream("Tell me a short story about a brave astronaut."):
        print(chunk, end="", flush=True)
    print("\n")

if **name** == "**main**":
from dotenv import load_dotenv

    load_dotenv()
    main()
    asyncio.run(async_ai_main())

import asyncio

from PIL import Image as PIL_Image
from pydantic import BaseModel, Field

from tiny_ai_client import AI, AsyncAI

class WeatherParams(BaseModel):
location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
unit: str = Field(
"celsius", description="Temperature unit", enum=["celsius", "fahrenheit"]
)

def get_current_weather(weather: WeatherParams):
"""
Get the current weather in a given location
"""
return {
"abc": f"Getting the current weather in {weather.location} in {weather.unit}."
}

def get_images():
return [PIL_Image.open("assets/kirk.jpg"), PIL_Image.open("assets/spock.jpg")]

def main():
print("### SYNC AI ###")
ai = AI(
model_name="gemini-1.5-flash",
system="You are Spock, from Star Trek.",
max_new_tokens=128,
tools=[],
)
response = ai("How are you?")
print(f"{response=}")
response = ai("Who is on the images?", images=get_images())
print(f"{response=}") # print(f"{ai.chat=}")

async def async_ai_main():
print("### ASYNC AI ###")
ai = AsyncAI(
model_name="gemini-1.5-flash",
system="You are Spock, from Star Trek.",
max_new_tokens=128,
tools=[],
)
response = await ai("How are you?")
print(f"{response=}")
response = await ai("Who is on the images?", images=get_images())
print(f"{response=}") # print(f"{ai.chat=}")

if **name** == "**main**":
from dotenv import load_dotenv

    load_dotenv()
    main()
    asyncio.run(async_ai_main())[0m

[1mπ [0m```sh
cat > examples/gemini\_.py << EOL
import asyncio

from PIL import Image as PIL_Image
from pydantic import BaseModel, Field

from tiny_ai_client import AI, AsyncAI

class WeatherParams(BaseModel):
location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
unit: str = Field(
"celsius", description="Temperature unit", enum=["celsius", "fahrenheit"]
)

def get_current_weather(weather: WeatherParams):
"""
Get the current weather in a given location
"""
return {
"abc": f"Getting the current weather in {weather.location} in {weather.unit}."
}

def get_images():
return [PIL_Image.open("assets/kirk.jpg"), PIL_Image.open("assets/spock.jpg")]

def main():
print("### SYNC AI ###")
ai = AI(
model_name="gemini-1.5-flash",
system="You are Spock, from Star Trek.",
max_new_tokens=128,
tools=[],
)
response = ai("How are you?")
print(f"{response=}")
response = ai("Who is on the images?", images=get_images())
print(f"{response=}") # print(f"{ai.chat=}")

    print("\n### SYNC AI STREAMING ###")
    for chunk in ai.stream("Tell me a short story about a brave astronaut."):
        print(chunk, end="", flush=True)
    print("\n")

async def async_ai_main():
print("### ASYNC AI ###")
ai = AsyncAI(
model_name="gemini-1.5-flash",
system="You are Spock, from Star Trek.",
max_new_tokens=128,
tools=[],
)
response = await ai("How are you?")
print(f"{response=}")
response = await ai("Who is on the images?", images=get_images())
print(f"{response=}") # print(f"{ai.chat=}")

    print("\n### ASYNC AI STREAMING ###")
    async for chunk in ai.astream("Tell me a short story about a brave astronaut."):
        print(chunk, end="", flush=True)
    print("\n")

if **name** == "**main**":
from dotenv import load_dotenv

    load_dotenv()
    main()
    asyncio.run(async_ai_main())

EOL

```
[31mPress enter to execute, or 'N' to cancel.[0m
[F[K[F[K[2m[0m
[1mπ [0m
```
