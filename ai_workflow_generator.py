from langchain_core.output_parsers import JsonOutputParser

from tools.ai_workflow_generator_tools import *


class AiWorkflowGenerator:
    def __init__(self):
        self.ai_manager = _get_ai_manager()

    def generate(self, message: str):
        return self.ai_manager.answer(
            message,
            '''
            You are workflow generator. 
            Your job is parsing input through its dedicated tools 
            and then combine their outputs into full workflow.
            Answer only with generated JSON content.

            Examples:
            - Input: Search "What is space" in internet.
              Output:
              "steps": {
                "keyword": {
                  "type": "user_input",
                  "value": "What is space?"
                },
                "search_keyword": {
                  "type": "perform_internet_search",
                  "query": "{steps>keyword>result}",
                  "pages": 1
                }
              }
              Context: In this example, step "keyword" is result from calling tool "user_input_step"
              and "search_keyword" is result from calling tool "perform_internet_search".
              Tool "keyword" was called and and there is invisible property "result" which is passed to the tool
              "search_keyword" query using some kind of expression.
            - Input: Search "What is space" in internet and extract 3 first urls.
              Output:
              "steps": {
                "keyword": {
                  "type": "user_input",
                  "value": "What is space?"
                },
                "search_keyword": {
                  "type": "perform_internet_search",
                  "query": "{steps>keyword>result}",
                  "pages": 1
                },
                "extract_urls":{
                  "type": "extract_data_from_text",
                  "text": "{steps>search_keyword>result}",
                  "result_schema": [
                    {
                      "name": "first_url",
                      "description": "First extracted url"
                    },
                    {
                      "name": "second_url",
                      "description":"Second extracted url"
                    },
                    {
                      "name": "third_url",
                      "description": "Third extracted url"
                    }
                  ]
                }
              }
            '''
        )


def _get_ai_manager() -> AiManager:
    ai_manager = AiManager(temperature=0)
    ai_manager.add_tools([
        user_input_step,
        perform_internet_search_step,
        extract_data_from_text_step,
    ])
    ai_manager.set_parser(JsonOutputParser())

    return ai_manager
