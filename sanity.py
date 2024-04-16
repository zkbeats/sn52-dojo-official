import copy
import random
import pandas as pd
import jsonref
from pydantic import BaseModel
import json
from openai import OpenAI


# gpt-3.5
from os import getenv
import os
import traceback
from typing import List
from dotenv import load_dotenv


load_dotenv()

generate_coding_question = """
Generate a short, self-contained, challenging coding problem that requires the programmer to output an visualization from the piece of code with 3 requirements on the functionality of the interactions. The interactions must require the programmer to have a mental model of any objects being visualized. You must also provide valid test case for the programmer to validate their code. the programmer is allowed to code using python or javascript or pure html.
"""


# Define your desired output structure
class CodingQuestion(BaseModel):
    title: str
    description: str
    language: str


class CodeAnswer(BaseModel):
    code: str
    language: str
    installation_commands: str


# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=getenv("OPENROUTER_API_KEY"),
)


# Function to call the API multiple times and save the data to a jsonl file
def generate_questions_and_save_to_file(client, num_questions):
    questions = []
    for _ in range(num_questions):
        qn = client.chat.completions.create(
            model="openai/gpt-4-turbo",
            response_format=PydanticUtils.build_response_format(CodingQuestion),  # noqa: F821
            messages=[{"role": "system", "content": generate_coding_question}],
            # seed=random.randint(0, 1000000),
        )
        print(f"Got question: {qn.dict()}")
        questions.append(qn.dict())

    with open("questions.jsonl", "w") as outfile:
        for question in questions:
            json.dump(question, outfile)
            outfile.write("\n")


sampled_models = [
    # "phind/phind-codellama-34b",
    # "anthropic/claude-3-opus",
    # "codellama/codellama-70b-instruct",
    "cognitivecomputations/dolphin-mixtral-8x7b",
]


def remove_key(input_dict, key, depth=0):
    """Recursively remove a specified key from a nested dictionary, keeping track of depth."""
    for k, v in list(input_dict.items()):
        if k == key:
            del input_dict[k]
        elif isinstance(v, dict):
            remove_key(v, key, depth=depth + 1)
    return input_dict


def _resolve_references(json_str):
    return jsonref.loads(json_str)


class PydanticUtils:
    @staticmethod
    def build_response_format(model: BaseModel):
        """Build a response format for OpenAI API calls."""
        schema = model.schema_json()
        resolved_schema = copy.deepcopy(_resolve_references(schema))

        if "definitions" in resolved_schema:
            resolved_schema.pop("definitions")

        resolved_schema = remove_key(resolved_schema, "title")
        resolved_schema = remove_key(resolved_schema, "additionalProperties")
        required = resolved_schema.get("required", [])
        resolved_schema = remove_key(resolved_schema, "required")
        resolved_schema["required"] = required
        return {"type": "json_object", "schema": resolved_schema}


def create_prompt(title, description):
    return f"""Your task is to solve the following coding task and provide your code solution directly, by only using built-in libraries.
You must assume that you do not have access to the file system, therefore if any test data is provided, you must store it in memory appropriately in some variable(s) and not in a file.
You must not provide any other text or explanations.
Additionally, your code solution needs to directly execute any functions required to provide the solution to the task.
You must not use any external libraries or packages.

Title: {title}, Task: {description}.

Code solution:
"""


def generate_solutions():
    # Load the data into a DataFrame
    questions_df = pd.read_json("questions.jsonl", lines=True)
    print("Data loaded into DataFrame.")
    print(PydanticUtils.build_response_format(CodeAnswer))

    for index, row in questions_df.iterrows():
        for model in sampled_models:
            response = client.chat.completions.create(
                model=model,
                # response_format=PydanticUtils.build_response_format(CodeAnswer),
                messages=[
                    {
                        "role": "system",
                        "content": create_prompt(row["title"], row["description"]),
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            print(f"Got OpenAI response: {response.choices[0]=}")
            try:
                content = response.choices[0].message.content
                ans = CodeAnswer.parse_raw(content)
                print(f"Got answer: {ans=}")
                questions_df.at[index, model] = ans.code
            except:
                traceback.print_exc()
                pass
            else:
                questions_df.to_json(
                    "updated_questions.jsonl", orient="records", lines=True
                )
                print("Updated DataFrame saved to updated_questions.jsonl.")


if __name__ == "__main__":
    if not os.path.exists("questions.jsonl"):
        generate_questions_and_save_to_file(client, num_questions=10)

    # generate_solutions()
