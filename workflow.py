from resolvers import *


class Runner:
    def __init__(self, context: dict):
        self.context = context
        self.resolvers = get_resolvers()

    def run(self):
        count = 0
        for k, step in self.context['steps'].items():
            count += 1
            resolver = self.get_resolver(step['type'])
            result = resolver.resolve(step, self.context)
            self.context['steps'][k]['result'] = result

            print(str(count) + '.')
            # print(result)
            if resolver.get_type() == 'scrape_webpage':
                print('Scrapping webpage completed.')
            else:
                print(result)

        return self.context

    def get_resolver(self, type: str) -> Resolver:
        for resolver in self.resolvers:
            if resolver.get_type() == type:
                return resolver

        raise Exception('No resolver found for type {}'.format(type))
