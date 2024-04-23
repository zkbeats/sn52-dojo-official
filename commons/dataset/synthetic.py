import asyncio
from asyncio.log import logger
import copy
import json
from logging import log
import random

# regex output
import re
import textwrap
import traceback
from typing import List, Optional
from openai import AsyncOpenAI, OpenAIError
import openai

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed
import tenacity
from commons.custom_exceptions import MaximumRetriesReached

from commons.llm.openai_proxy import Provider, get_openai_client
from commons.utils import PydanticUtils, log_retry_info
import bittensor as bt
from openai.types.chat import ChatCompletion

load_dotenv()


class CodingQuestion(BaseModel):
    question: str = Field(
        description="Coding question to be solved by a software engineer"
    )
    languages: List[str] = Field(
        description="Allowed programming languages for the programmer to use"
    )


class CodeAnswer(BaseModel):
    code: str = Field(description="Code solution to the question")
    language: str = Field(description="Programming language of the code")
    installation_commands: str = Field(
        description="Terminal commands for the code to be able to run to install any third-party packages for the code to be able to run"
    )
    additional_notes: Optional[str] = Field(
        description="Any additional notes or comments about the code solution"
    )


def build_code_generation_question_prompt(
    num_requirements: int = random.choices([3, 4, 5], weights=[0.5, 0.3, 0.2])[0],
) -> str:
    bt.logging.info(f"Generating question with {num_requirements} requirements")
    CODE_GEN_PROMPT = """
    System:
    - Generate a short, self-contained, challenging coding problem that requires the programmer to output an visualization from the piece of code with {num_requirements} requirements on the functionality of the interactions.
    - The interactions must require the programmer to have a mental model of any objects being visualized.
    - The question generated must require the programmer to code using only Python, or Javascript with HTML and CSS.
    - You must not provide any example code snippets, because you must let the programmer solve the question by themselves.
    - If the generated question is in Python, it should command the usage of built-in libraries or third-party visualization libraries like plotly, matplotlib and tkinter. You must discourage the usage of libraries like pygame.
    - If the generated question is in Javascript, it should command the usage of built-in libraries or use visualization libraries like three.js, D3.js.

    Coding Question:
    """
    return textwrap.dedent(CODE_GEN_PROMPT.format(num_requirements=num_requirements))


def build_code_augmenter_prompt() -> str:
    pass


def parse_openai_json_mode_response(completion_content: str):
    pydantic_utils_keys = ["type", "schema"]
    parsed = None
    try:
        json_content = extract_json(completion_content)
        if json_content:
            # successfully extracted json content
            parsed = json.loads(json_content)
        else:
            parsed = json.loads(completion_content)
    except json.JSONDecodeError as e:
        bt.logging.info(f"Error occurred while parsing JSON response: {e}")
    except Exception as e:
        pass

    if parsed:
        for key in pydantic_utils_keys:
            if key in parsed:
                parsed.pop(key)
    return parsed


def build_code_answer_prompt(question) -> str:
    CODE_ANS_PROMPT = """
    System:
    - Your task is to solve the coding question below, according to the fields in the JSON_SCHEMA: {json_schema}.
    - You must assume that you do not have access to the file system, therefore if any test data is provided, you must store it in memory appropriately in the necessary variable and not in a file.
    - You must not provide any other text or explanations.
    - You must provide all code required to ensure that your solution is complete.
    - Do not leave out any details for brevity.
    - Additionally, ensure that your code solution directly executes any functions required to provide the solution to the task.

    Question:
    {question}

    Answer according to the JSON_SCHEMA:
    """

    return textwrap.dedent(
        CODE_ANS_PROMPT.format(
            json_schema=PydanticUtils.build_response_format(CodeAnswer),
            question=question,
        )
    )


# # Function to call the API multiple times and save the data to a jsonl file
# def generate_questions_and_save_to_file(client, num_questions):
#     questions = []
#     for _ in range(num_questions):
#         qn = client.chat.completions.create(
#             model="openai/gpt-4-turbo",
#             response_format=PydanticUtils.build_response_format(CodingQuestion),  # noqa: F821
#             messages=[{"role": "system", "content": CODE_GEN_PROMPT}],
#             # seed=random.randint(0, 1000000),
#         )
#         print(f"Got question: {qn.dict()}")
#         questions.append(qn.dict())

#     with open("questions.jsonl", "w") as outfile:
#         for question in questions:
#             json.dump(question, outfile)
#             outfile.write("\n")


# def generate_solutions(sampled_models: List[str]):
#     questions_df = pd.read_json("questions.jsonl", lines=True)
#     print("Data loaded into DataFrame.")
#     print(PydanticUtils.build_response_format(CodeAnswer))

#     for index, row in questions_df.iterrows():
#         for model in sampled_models:
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": build_code_answer_prompt(
#                             row["title"], row["description"]
#                         ),
#                     },
#                 ],
#                 temperature=0.1,
#                 max_tokens=4096,
#             )
#             print(f"Got OpenAI response: {response.choices[0]=}")
#             try:
#                 content = response.choices[0].message.content
#                 ans = CodeAnswer.parse_raw(content)
#                 print(f"Got answer: {ans=}")
#                 questions_df.at[index, model] = ans.code
#             except:
#                 traceback.print_exc()
#                 pass
#             else:
#                 questions_df.to_json(
#                     "updated_questions.jsonl", orient="records", lines=True
#                 )
#                 print("Updated DataFrame saved to updated_questions.jsonl.")


# if not os.path.exists("questions.jsonl"):
#     generate_questions_and_save_to_file(client, num_questions=10)

# questions_list = []
# with open("questions.jsonl", "r") as f:
#     for line in f:
#         questions_list.append(json.loads(line))

# for question in questions_list:
#     print(question["choices"][0]["message"]["content"])


# def extract_code(raw: str):
#     pattern = r"```(?:[^\n]*\n)?(.*?)```"
#     extracted_code = re.search(pattern, raw, re.DOTALL)

#     if extracted_code:
#         code_block = extracted_code.group(1)
#         # print("Extracted code block:")
#         # print(code_block)
#         return code_block
#     else:
#         print("No code block found between triple backticks.")
#         return None


def extract_json(text) -> Optional[str]:
    """Returns anything from between ```json ```"""
    pattern = r"(?<=\`\`\`json\n)([\s\S]*?)(?=\n\`\`\`)"
    extracted_code = re.search(pattern, text, re.DOTALL)
    if extracted_code:
        code_block = extracted_code.group(1)
        return code_block
    else:
        print("No code block found between triple backticks.")
        return None


def extract_code_from_response(response):
    # call gpt4 to rephrase
    extracted = client.chat.completions.create(
        model="openai/gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": f"Please provide the plain code block from the following without any formatting or syntax markers like triple backticks. \n{response}",
            }
        ],
        temperature=0.0,
        max_tokens=8192,
    )
    return extracted.choices[0].message.content


async def generate_question(client: AsyncOpenAI, model: str) -> Optional[str]:
    MAX_RETRIES = 10
    kwargs = {
        "model": model,
        # num requirements get randomized here
        "messages": [
            {
                "role": "system",
                "content": build_code_generation_question_prompt(),
            }
        ],
        "temperature": 0.2,
        "max_tokens": 4096,
        "top_p": random.uniform(0.9, 1.0),
    }

    if model.startswith("openai"):
        kwargs["seed"] = random.randint(0, 1e9)

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_fixed(0.10),
            before_sleep=log_retry_info,
        ):
            with attempt:
                completion = await client.chat.completions.create(**kwargs)
                coding_question = completion.choices[0].message.content
                bt.logging.info(f"Generated question: {coding_question}")
                # attempt.retry_state.attempt_number
                return coding_question
    except RetryError:
        bt.logging.error(
            f"Failed to generate completion after {MAX_RETRIES} attempts while evaluating human preference.",
        )
        # raise MaximumRetriesReached(
        #     f"Maximum retries of {MAX_RETRIES} reached, and failed to genrate question."
        # )
        pass

    return None


def on_error_update_kwargs(completion: ChatCompletion, kwargs_dict: dict):
    if not hasattr(completion, "error"):
        # false to tell caller kwargs weren't updated
        return False, kwargs_dict

    error_msg = completion.error.get("message") if completion.error else None
    if error_msg and "invalid_request_error" in error_msg and "not supported for JSON":
        kwargs_dict.pop("response_format")

    # kwargs were updated
    return True, kwargs_dict


async def generate_answer(client: AsyncOpenAI, model: str, question: str):
    """Generates a coding question answer for a given coding question."""
    MAX_RETRIES = 10
    kwargs = {
        "model": model,
        "response_format": {
            "type": "json_object",
            "schema": PydanticUtils.build_response_format(CodeAnswer),
        },
        "temperature": 0.0,
        "max_tokens": 8192,
        "messages": [
            {
                "role": "system",
                "content": build_code_answer_prompt(question),
            },
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        ],
    }
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_fixed(0.10),
            before_sleep=log_retry_info,
        ):
            with attempt:
                completion = await client.chat.completions.create(**kwargs)
                bt.logging.warning(f"Completion: {completion}")
                is_updated, kwargs = on_error_update_kwargs(completion, kwargs)
                if is_updated:
                    bt.logging.warning(
                        f"Updated kwargs due to error, {completion.error}"
                    )

                if completion.choices is None:
                    bt.logging.error(
                        f"No choices found in completion for model {model}"
                    )
                content_json = parse_openai_json_mode_response(
                    completion.choices[0].message.content
                )
                completion_content = completion.choices[0].message.content
                bt.logging.warning(f"Completion content: {completion_content}")
                bt.logging.warning(f" content json: {content_json}")

                code_answer = CodeAnswer.parse_obj(content_json)
                return model, code_answer
    except RetryError:
        bt.logging.error(
            f"Failed to generate completion after {MAX_RETRIES} attempts for generating code answer"
        )
    return model, None


# df = pd.DataFrame(columns=["question"] + sampled_models)
# async def get_model_response(client, model, prompt):
#     response = None
#     print(f"Getting response for model {model}")
#     try:
#         response = await client.chat.completions.create(
#             model=model,
#             response_format=PydanticUtils.build_response_format(CodeAnswer),
#             messages=[
#                 {"role": "system", "content": prompt},
#             ],
#             temperature=0.0,
#             max_tokens=4096,
#         )
#         # no error occurred... try to extract
#         extracted_response = await client.chat.completions.create(
#             model="openai/gpt-4-turbo",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"Given following JSON schema, {PydanticUtils.build_response_format(CodeAnswer)}, extract information from the provided text below to create a JSON object.\nProvided Text: {response.choices[0].message.content} JSON Object:",
#                 }
#             ],
#             temperature=0,
#             max_tokens=1.1 * 4096,
#         )
#         print(f"Got parsed response: {extracted_response.choices[0].message.content}")
#         try:
#             json_data = json.loads(extracted_response.choices[0].message.content)
#             return json_data
#         except json.JSONDecodeError:
#             json_data = json.loads(
#                 extract_json(extracted_response.choices[0].message.content)
#             )
#             return json_data
#     except Exception as e:
#         print(f"Error occurred for model {model}: {e}")
#         traceback.print_exc()
#         return None


# async def process_question(client, question, sampled_models):
#     print("Sampling models: ", sampled_models)
#     prompt = build_code_answer_prompt(question["choices"][0]["message"]["content"])
#     responses = await asyncio.gather(
#         *[get_model_response(client, model, prompt) for model in sampled_models]
#     )
#     question_responses = {"question": question["choices"][0]["message"]["content"]}

#     question_responses["responses"] = []
#     for response, model in zip(responses, sampled_models):
#         # if response is None or not response.choices:
#         if response is None:
#             print(f"No response or no choices for model: {model}")
#             continue
#         try:
#             answer = CodeAnswer.parse_obj(response)
#             curr_response = {
#                 **answer.dict(),
#                 "model": model,
#             }
#             if "code" in curr_response:
#                 curr_response["code"] = json.dumps(curr_response["code"])
#             question_responses["responses"].append(curr_response)
#         except:
#             print(f"Error occurred while parsing response for model {model}")
#             traceback.print_exc()

#     return question_responses


# # Run the main coroutine and wait for it to finish
# q_responses_pair = await main(client, questions_list[:1], sampled_models)

# # Display the DataFrame
# print(q_responses_pair)

# q_responses_pair["responses"]

# # save dataframe to jsonl file
# q_responses_pair.to_json("model_responses.jsonl", orient="records", lines=True)

# # # rating outputs
# # - grab all outputs for a certain question, and ask GPT-4 to score them
# # - obfuscate the code example of eahc answer, then try asking gpt-4 again
# # - compare non-obfuscated rating vs obfuscated ratings


# # Parse each response's code field using json.loads
# for response in q_responses_pair["responses"]:
#     response["code"] = json.loads(response["code"])


# q_responses_pair["responses"]

#### RATING NON-OBFUSCATED CODE
#### RATING NON-OBFUSCATED CODE
#### RATING NON-OBFUSCATED CODE
#### RATING NON-OBFUSCATED CODE
#### RATING NON-OBFUSCATED CODE

# q_responses_pair["responses"]


def create_eval_prompt(question, response):
    return f"""
System: Your task is to evaluate a code snippet that is a solution to the coding problem, and provide a score between 0 and 10 based on the following criteria: correctness, efficiency, readability, and maintainability. Follow the following format for the response:
{{"score": "<score between 0 to 10", "reasoning": "<reasoning for the score>"}}

Problem Statement:
{question}

Solution:
{response['code']}

Score: Let's think step by step.
"""


# async def rate_output(question, response: dict, judge_model):
#     completion = await client.chat.completions.create(
#         model=judge_model,
#         messages=[
#             {"role": "system", "content": create_eval_prompt(question, response)},
#         ],
#         temperature=0.0,
#         max_tokens=4096,
#     )
#     try:
#         score = completion.choices[0].message.content
#         return score
#     except:
#         traceback.print_exc()
#         return None


# for response in q_responses_pair["responses"]:
#     score = await rate_output(q_responses_pair["question"], response, judge_model)
#     model = response["model"]
#     print(f"Model: {model}, Score: {score}")


# # obfuscate the code using calmjs.parse
# def obfuscate_js(code):
#     from calmjs.parse.unparsers.es5 import minify_print

#     return minify_print(code, obfuscate=True, obfuscate_globals=True)


# q_responses_pair_copy = copy.deepcopy(q_responses_pair)

# q_responses_pair_copy["responses"]

# obfuscate_js(q_responses_pair_copy["responses"][1]["code"])

# from jsmin import jsmin

# minified = jsmin(q_responses_pair_copy["responses"][1]["code"])

# minified

# # obfuscate html
# from html_classes_obfuscator import html_classes_obfuscator


# def generate_class(current_classes_list):
#     def random_class():
#         # Offers (26*2)^6 random class name possibilities
#         return "".join(random.choice(string.ascii_letters) for i in range(6))

#     res = random_class()

#     while res in current_classes_list.values():
#         res = random_class()

#     return res


# import tempfile

# with tempfile.NamedTemporaryFile("w+t") as temp_file:
#     temp_file.write(q_responses_pair_copy["responses"][1]["code"])
#     temp_file.seek(0)
#     path = temp_file.name
#     html_classes_obfuscator.html_classes_obfuscator([path], [], [], generate_class)
#     # Print the contents of the HTML file after obfuscation
#     obfuscated_html_content = temp_file.read()
#     print(obfuscated_html_content)


# from slimit import minify

# minify(q_responses_pair_copy["responses"][1]["code"], mangle=True)


async def build_prompt_responses_pair():
    client = get_openai_client(Provider.OPENROUTER)
    # use these models because we can specify seed
    generator_models = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-3.5-turbo-0613",
        "openai/gpt-3.5-turbo-0301",
        "openai/gpt-3.5-turbo-16k",
        "openai/gpt-4-turbo",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4-1106-preview",
        "openai/gpt-4",
        "openai/gpt-4-0314",
        "openai/gpt-4-32k",
        "openai/gpt-4-32k-0314",
        "openai/gpt-4-vision-preview",
        "openai/gpt-3.5-turbo-instruct",
    ]

    # tasks = [generate_question(client, model) for model in generator_models]
    # questions = await asyncio.gather(*tasks)
    prompt = await generate_question(client, random.choice(generator_models))

    # NOTE @dev LLMs here were selected to be able to compare against the EvalPLus leaderboard
    answer_models = [
        "phind/phind-codellama-34b-v2",
        "microsoft/wizardlm-2-8x22b",
        "gpt-4-turbo-2024-04-09",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-haiku-20240307",
        "codellama/codellama-70b-instruct",
        "mistralai/mistral-large",
        "google/gemini-pro-1.5",
        "cognitivecomputations/dolphin-mixtral-8x7b",
        "cohere/command-r-plus",
        "google/gemini-pro-1.0",
        "meta-llama/llama-3-8b-instruct",
    ]
    # select 2 of 3 answer models
    sel_ans_models = random.sample(answer_models, 4)

    results = await asyncio.gather(
        *[generate_answer(client, ans_model, prompt) for ans_model in sel_ans_models]
    )

    for model, answer in results:
        print(f"Got answer for model: {model}, answer: {answer}")


async def main():
    judge_model = "openai/gpt-4-turbo"
    await build_prompt_responses_pair()


if __name__ == "__main__":
    asyncio.run(main())
