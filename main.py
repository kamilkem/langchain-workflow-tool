import json

from workflow import Runner
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

with open("workflow.json", "r") as file:
    config = json.load(file)

runner = Runner(config)
result = runner.run()

# with open('result.json', 'w') as file:
#     json.dump(result, file)
