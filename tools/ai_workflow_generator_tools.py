import uuid
from typing import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from ai_manager import AiManager
from models import *


class Example(TypedDict):
    """
    A representation of an example consisting of text input and expected tool calls.
    For extraction, the tool calls are represented as instances of pydantic model.
    """
    input: str
    tool_calls: List[BaseModel]


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example['input'])]
    openai_tool_calls = []
    for tool_call in example['tool_calls']:
        openai_tool_calls.append(
            {
                'id': str(uuid.uuid4()),
                'type': 'function',
                'function': {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    'name': tool_call.__class__.__name__,
                    'arguments': tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content='', additional_kwargs={'tool_calls': openai_tool_calls})
    )
    tool_outputs = example.get('tool_outputs') or [
        'You have correctly called this tool.'
    ] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call['id']))
    return messages


@tool
def user_input_step(message: str) -> str:
    """
    You are workflow step generator. Your job is to generate specific step schema for a bigger workflow.
    Use already provided information to generate UserInputStep schema.
    """
    ai_manager = AiManager(temperature=0)
    ai_manager.set_pydantic_schema(UserInputStep)
    answer = ai_manager.answer(
        message,
        '''
        You are workflow step generator. Your job is to generate specific step schema for a bigger workflow.
        Use already provided information to generate UserInputStep schema.
        '''
    )
    print({'name': 'user_input_step', 'message': message, 'answer': answer})

    return answer


@tool
def perform_internet_search_step(query: str) -> str:
    """
    You are workflow step generator. Your job is to generate specific step schema for a bigger workflow.Wh
    Use provided information to generate InternetSearchStep schema.
    """
    ai_manager = AiManager(temperature=0)
    ai_manager.set_pydantic_schema(PerformInternetSearchStep)
    answer = ai_manager.answer(
        query,
        '''
        You are workflow step generator. Your job is to generate specific step schema for a bigger workflow.
        Use already provided information to generate UserInputStep schema.
        '''
    )
    print({'name': 'perform_internet_search_step', 'query': query, 'answer': answer})

    return answer


@tool
def extract_data_from_text_step(text: str) -> str:
    """
    Use this tool when user asks for extracting some data.
    You are workflow step generator. Your job is to generate specific step schema for a bigger workflow.
    Use already provided information to generate ExtractDataFromTextStep schema.
    """
    text="Extract top 3 urls"
    print(text)
    ai_manager = AiManager(temperature=0)
    ai_manager.set_pydantic_schema(ExtractDataFromTextStep)
    answer = ai_manager.answer(
        text,
        '''
        You are workflow step generator. Your job is to generate specific step schema for a bigger workflow.
        Use already provided information to generate ExtractDataFromTextStep schema.
        '''
    )
    print({'name': 'extract_data_from_text_step', 'text': text, 'answer': answer})

    return answer
