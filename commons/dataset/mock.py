import random
from typing import List
from template.protocol import FeedbackRequest, Completion, TaskType, CriteriaType


class MockData:
    all_models = [
        "mistralai/mixtral-8x22b-instruct",
        "openai/gpt-4-turbo-2024-04-09",
        "openai/gpt-4-1106-preview",
        "openai/gpt-3.5-turbo-1106",
        "meta-llama/llama-3-70b-instruct",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-haiku-20240307",
        "mistralai/mistral-large",
        "google/gemini-pro-1.5",
        "cognitivecomputations/dolphin-mixtral-8x7b",
        "cohere/command-r-plus",
        "google/gemini-pro-1.0",
        "meta-llama/llama-3-8b-instruct",
    ]

    @classmethod
    def generate_mock_data(all_models: List[str]):
        test_requests = []
        model_ids = random.sample(all_models, 4)
        for i in range(1, 11):
            ranks = list(range(1, 5))
            random.shuffle(ranks)
            random.shuffle(model_ids)
            test_requests.append(
                FeedbackRequest(
                    request_id=f"req{i}",
                    prompt=f"Prompt for request {i}",
                    completions=[
                        Completion(
                            cid=f"c{i}1",
                            model_id=model_ids[0],
                            text=f"Text {i}1",
                            rank_id=ranks[0],
                            code="",
                            language="deez",
                            installation_commands="",
                        ),
                        Completion(
                            cid=f"c{i}2",
                            model_id=model_ids[1],
                            text=f"Text {i}2",
                            rank_id=ranks[1],
                            code="",
                            language="deez",
                            installation_commands="",
                        ),
                        Completion(
                            cid=f"c{i}3",
                            model_id=model_ids[2],
                            text=f"Text {i}3",
                            rank_id=ranks[2],
                            code="",
                            language="deez",
                            installation_commands="",
                        ),
                        Completion(
                            cid=f"c{i}4",
                            model_id=model_ids[3],
                            text=f"Text {i}4",
                            rank_id=ranks[3],
                            code="",
                            language="deez",
                            installation_commands="",
                        ),
                    ],
                    scoring_method="dojo_worker",
                    task_type=TaskType.CODE_GENERATION,
                    criteria_types=[CriteriaType.PREFERENCE_RANKING],
                )
            )
        return test_requests
