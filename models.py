from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field


class ExtractDataFromTextStepResultSchema(BaseModel):
    name: str = Field('Key value of extracted data')
    description: str = Field('Description of extracted data')


class Step(BaseModel):
    def get_type(self) -> str:
        pass


class UserInputStep(Step):
    """User input step schema"""

    def get_type(self) -> str:
        return 'user_input'

    value: str = Field(default=None, description='The user input value.')


class PerformInternetSearchStep(Step):
    """Perform internet search step schema"""

    def get_type(self) -> str:
        return 'perform_internet_search'

    query: str = Field(default=None, description='The search query.')
    pages: int = Field(default=10, description='Number of searched pages. One page provides 10 results.')


class ExtractDataFromTextStep(Step):
    """Extract data from text step schema"""

    def get_type(self) -> str:
        return 'extract_data'

    text: str = Field(default=None, description="Text to extract data from.")
    result_schema: str = Field(default=None, description="Generated schema based on text to extract.")
