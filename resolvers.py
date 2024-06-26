import re

from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_manager import AiManager


class Resolver:
    def resolve(self, step: dict, context: dict):
        pass

    def get_type(self) -> str:
        pass


class UserInputResolver(Resolver):
    def resolve(self, step: dict, context: dict):
        return step['value']

    def get_type(self) -> str:
        return 'user_input'


class ScrapeWebpageResolver(Resolver):
    def resolve(self, step: dict, context: dict):
        loader = AsyncChromiumLoader([enrich_string_with_context(step['url'], context)])
        html = loader.load()

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=15000)
        splits = splitter.split_documents(html)

        return splits[0]

    def get_type(self) -> str:
        return 'scrape_webpage'


class AiGenerateTextResolver(Resolver):
    def resolve(self, step: dict[str], context: dict):
        model_name = str(enrich_string_with_context('{defaults>model_name}', context))
        temperature = float(enrich_string_with_context('{defaults>temperature}', context))

        manager = AiManager(model_name=model_name, temperature=temperature)
        input = enrich_string_with_context(step['human'], context)

        return manager.answer(input, step['system'])

    def get_type(self) -> str:
        return 'ai_generate_text'


class PerformInternetSearchResolver(Resolver):
    def resolve(self, step: dict, context: dict):
        model_name = str(enrich_string_with_context('{defaults>model_name}', context))
        temperature = float(enrich_string_with_context('{defaults>temperature}', context))

        manager = AiManager(model_name=model_name, temperature=temperature)
        manager.add_search_tool()
        input = enrich_string_with_context(step['query'], context)
        pages = step['pages']

        return manager.answer(
            'Query: ' + input,
            'You are like web browser. '
            'Your job is to present links from results from provided using search tool. '
            f'Give exactly {pages * 10} results.'
        )['output']

    def get_type(self) -> str:
        return 'perform_internet_search'


class ExtractDataFromTextResolver(Resolver):
    def resolve(self, step: dict, context: dict):
        model_name = str(enrich_string_with_context('{defaults>model_name}', context))
        temperature = float(0)

        manager = AiManager(model_name=model_name, temperature=temperature)
        manager.add_response_schema(step['schema'])
        input = enrich_string_with_context(step['text'], context)

        return manager.answer(
            'Text to extract from: ' + input,
            'You are an expert extraction algorithm. '
            'Only extract relevant information from the text.'
            'If you do not know the value of an attribute asked '
            'to extract, return null for the attribute\'s value.'
        )

    def get_type(self) -> str:
        return 'extract_data_from_text'


def get_resolvers() -> list[Resolver]:
    return [
        UserInputResolver(),
        ScrapeWebpageResolver(),
        AiGenerateTextResolver(),
        PerformInternetSearchResolver(),
        ExtractDataFromTextResolver(),
    ]


def enrich_string_with_context(string: str, context: dict) -> str:
    extracted_properties = re.findall('\{([^}]+)\}', string)
    enriched_properties = {}
    for prop in extracted_properties:
        keys = prop.split('>')
        result = context
        for key in keys:
            result = result[key]
        enriched_properties[prop] = result

    return string.format(**enriched_properties)
