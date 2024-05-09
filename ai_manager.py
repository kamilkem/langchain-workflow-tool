from langchain.agents import AgentExecutor, Tool, create_tool_calling_agent
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


class AiManager:
    def __init__(self, model_name: str, temperature: float):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._get_llm_model()
        self.tools = []
        self.response_schemas = []

    def _get_llm_model(self):
        return ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

    def _get_executor(self):
        prompt = None
        output_parser = None

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
        if not output_parser:
            return prompt | self.llm

        return prompt | self.llm | output_parser

    def add_search_tool(self):
        search = SerpAPIWrapper()
        self.tools.append(Tool(
            name='Search',
            func=search.run,
            description='Useful for searching information from internet.'
        ))

    def add_schema(self, schema: list):
        for item in schema:
            self.response_schemas.append(ResponseSchema(name=item['name'], description=item['description']))

    def answer(self, human: str, system: str):
        executor = self._get_executor()
        return executor.invoke({
            'human': human,
            'system': system
        })
