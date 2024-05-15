from typing import Type, Optional

from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Output
from langchain_openai import ChatOpenAI


class AiManager:
    def __init__(self, model_name: str = 'gpt-3.5-turbo-1106', temperature: float = 0.7):
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.llm: BaseChatModel = self._get_llm_model()
        self.tools: list = []
        self.response_schemas: list = []
        self.schema: Optional[Type[BaseModel]] = None
        self.parser: BaseOutputParser = StrOutputParser()

    def _get_llm_model(self) -> BaseChatModel:
        return ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

    def _get_executor(self) -> Runnable:
        prompt = None
        output_parser = self.parser

        if self.schema:
            prompt = ChatPromptTemplate.from_messages([
                ('system', '{system}'),
                ('human', '{human}')
            ])

            return prompt | self.llm.with_structured_output(schema=self.schema)

        if self.response_schemas:
            output_parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

            format_instructions = output_parser.get_format_instructions()
            prompt = PromptTemplate(
                template='{system}\n{format_instructions}\n{human}',
                input_variables=["human"],
                partial_variables={"format_instructions": format_instructions}
            )

        if self.tools:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', '{system}'),
                    ('human', '{human}'),
                    MessagesPlaceholder(variable_name='agent_scratchpad')
                ]
            )
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            return AgentExecutor(agent=agent, tools=self.tools)

        if not prompt:
            prompt = ChatPromptTemplate.from_messages([
                ('system', '{system}'),
                ('human', '{human}')
            ])

        return prompt | self.llm | output_parser

    def add_tools(self, tools: list):
        for tool in tools:
            self.tools.append(tool)

    def add_search_tool(self):
        search = SerpAPIWrapper()
        self.tools.append(Tool(
            name='Search',
            func=search.run,
            description='Useful for searching information from internet.'
        ))

    def add_response_schema(self, schema: list):
        for item in schema:
            self.response_schemas.append(ResponseSchema(name=item['name'], description=item['description']))

    def set_pydantic_schema(self, schema: Type[BaseModel]):
        self.schema = schema

    def set_parser(self, parser: BaseOutputParser):
        self.parser = parser

    def answer(self, human: str, system: str) -> Output:
        executor = self._get_executor()
        return executor.invoke({
            'human': human,
            'system': system
        })
