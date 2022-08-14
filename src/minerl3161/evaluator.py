from minerl3161.agent import BaseAgent


# TODO: write tests
class Evaluator:
    def __init__(self, env) -> None:
        self.env = env

    def evaluate(self, agent: BaseAgent, episodes: int) -> dict:
        return {}

    def create_media(self, agent: BaseAgent) -> dict:
        return {}
