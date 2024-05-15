from dotenv import load_dotenv, find_dotenv

from ai_manager import AiManager
from tools.prompt_tools import *
from ai_workflow_generator import AiWorkflowGenerator

load_dotenv(find_dotenv())

text = input('Text: ')

# ai_manager = AiManager()
# ai_manager.add_tools([delegate_to_human])
#
#
# print(ai_manager.answer(text, 'You are helpful assistant.'))

generator = AiWorkflowGenerator()
print(generator.generate(text)['output'])
