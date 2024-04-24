import asyncio
import functools
import json
import os
import random
import re
import textwrap
from typing import Any, Callable, List, Optional

import bittensor as bt
from unsync import unsync
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from strictjson import strict_json
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed

from commons.llm.openai_proxy import Provider, get_openai_client
from commons.utils import PydanticUtils, log_retry_info

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
    - If the generated question is in Python, it must use built-in libraries. The only third-party visualization library allowed is bokeh.
    - If the generated question is in Javascript, it should command the usage of built-in libraries or use visualization libraries like three.js, D3.js.

    Coding Question:
    """
    return textwrap.dedent(CODE_GEN_PROMPT.format(num_requirements=num_requirements))


def build_code_augmenter_prompt() -> str:
    # TODO
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
    except Exception:
        pass

    if parsed:
        for key in pydantic_utils_keys:
            if key in parsed:
                parsed.pop(key)
    return parsed


def detect_chars_until_first_word(text: str):
    pattern = r"^.*?(?=\b\w)"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group()


def parse_code_response(strictjson_response: dict[str, Any], model: str) -> dict:
    """ensures consistent format of 'code' key"""
    if "code" not in strictjson_response:
        # bt.logging.warning(f"{strictjson_response.keys()}")
        raise ValueError(f"No code key found in strictjson response for model: {model}")

    try:
        # using re.match to check the first character, is a letter a-z (case insensitive)
        code_text = strictjson_response["code"]
        if re.match(r"^[a-zA-Z]", code_text):
            detected_chars = detect_chars_until_first_word(code_text)
            is_all_same_char = (
                True if detected_chars and len(set(detected_chars)) == 1 else False
            )
            if (
                detected_chars
                and is_all_same_char
                and code_text.startswith(detected_chars)
                and code_text.endswith(detected_chars)
            ):
                code_text = code_text[len(detected_chars) : -len(detected_chars)]
                if len(code_text) > 0:
                    strictjson_response["code"] = code_text
    except Exception as e:
        pass

    # code = extract_strictjson_code(strictjson_response["code"])
    # if code:
    #     strictjson_response["code"] = code
    # return strictjson_response
    return strictjson_response


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


def extract_strictjson_code(text) -> Optional[str]:
    pattern = re.compile(r"```[a-zA-Z]+([\s\S]*?)```", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        return None

    parsed = "".join(matches)
    return parsed.lstrip().rstrip()


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


async def generate_question(client: AsyncOpenAI, model: str) -> Optional[str]:
    MAX_RETRIES = 10
    kwargs = {
        "model": model,
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
            f"Failed to generate completion after {MAX_RETRIES} attempts while generating question.",
        )
        pass

    return None


def on_error_update_kwargs(completion: ChatCompletion, kwargs_dict: dict):
    if not hasattr(completion, "error"):
        # false to tell caller kwargs weren't updated
        return False, kwargs_dict

    error_msg_json_str = completion.error.get("message") if completion.error else None
    error_code = completion.error.get("code") if completion.error else None
    # handle phind error
    # data = """{'message': '{"error":{"message":"Phind/Phind-CodeLlama-34B-v2 is not supported for JSON mode/function calling","type":"invalid_request_error","param":null,"code":"constraints_model"}}', 'code': 400}"""
    error_msg_json = {}
    try:
        if error_msg_json_str:
            error_msg_json = json.loads(error_msg_json_str)
            bt.logging.info(
                f"Got error code: {error_code} and error message: {error_msg_json}"
            )
            bt.logging.info("Successfully parsed json")
    except json.JSONDecodeError:
        pass
    # handle no JSON mode
    if (
        error_msg_json
        and "invalid_request_error" in error_msg_json_str["type"]
        # and "not supported for JSON"
        and error_code in [400, 422]
    ):
        kwargs_dict.pop("response_format")
        bt.logging.warning("Updated kwargs due to JSON mode not supported...")

    # kwargs were updated
    return True, kwargs_dict


def strictjson_llm_wrapper(system_prompt, user_prompt, model, args_dict):
    """A wrapper for the AsyncOpenAI LLM call that strictjson will use.
    Simply calls the unsync'd version of the async function and return the result.
    """
    return my_strict_json_llm_answer(
        system_prompt, user_prompt, model, args_dict
    ).result()


@unsync
async def my_strict_json_llm_answer(
    system_prompt: str, user_prompt: str, model: str, kwargs: dict = {}
):
    """Unsync'd version of an AsyncOpenAI call so we can call in inside of a synchronous context."""
    async_client = get_openai_client(Provider.OPENROUTER)
    result = await async_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **kwargs,
    )
    return result.choices[0].message.content


async def generate_strictjson_answer(sys, user, callable_llm: Callable):
    loop = asyncio.get_running_loop()
    # NOTE strict_json expects LLM call needs to be synchronous here
    func = functools.partial(
        strict_json,
        system_prompt=sys,
        user_prompt=user,
        output_format=PydanticUtils.build_minimal_json(CodeAnswer),
        llm=callable_llm,
    )
    result = await loop.run_in_executor(None, func)

    return result


async def generate_answer(client: AsyncOpenAI, model: str, question: str):
    """Generates a coding question answer for a given coding question."""
    MAX_RETRIES = 3

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_fixed(0.10),
            before_sleep=log_retry_info,
        ):
            with attempt:
                # NOTE trying strictjson LLMs in order to stabilise the parsing of code outputs
                callable_llm = functools.partial(
                    strictjson_llm_wrapper,
                    model=model,
                    args_dict={
                        "temperature": 0.0,
                        "max_tokens": 8192,
                    },
                )
                completion = await generate_strictjson_answer(
                    sys=build_code_answer_prompt(question),
                    user="Remember to provide the code solution according your previous instructions.",
                    callable_llm=callable_llm,
                )
                completion = parse_code_response(completion, model)

                # TODO parse the response because of weird triple backticks or quotes
                # try:
                #     parsed = parse_code_response(completion)
                #     return model, parsed
                # except Exception as e:
                #     bt.logging.warning(
                #         "Failed to parse & extract code between triple backticks, naively returning original completion."
                #     )
                #     pass

                return model, completion
    except RetryError:
        bt.logging.error(
            f"Failed to generate completion after {MAX_RETRIES} attempts for generating code answer for {model}"
        )
        pass

    return model, None


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

    # randomly sampled from pool of models
    IS_TEST = os.getenv("IS_TEST", False)
    num_samples = len(answer_models) if IS_TEST else 4
    sel_ans_models = random.sample(answer_models, num_samples)

    results = await asyncio.gather(
        *[generate_answer(client, ans_model, prompt) for ans_model in sel_ans_models]
    )
    res = {"prompt": prompt, "responses": []}
    for model, result in results:
        if not result:
            continue
        res["responses"].append(
            {
                "model": model,
                "completion": {
                    "code": result["code"],
                    "language": result["language"],
                    "installation_commands": result["installation_commands"],
                    "additional_notes": result["additional_notes"],
                },
            }
        )
    return res


async def main():
    res = await build_prompt_responses_pair()
    print(f"{res=}")


if __name__ == "__main__":
    asyncio.run(main())
