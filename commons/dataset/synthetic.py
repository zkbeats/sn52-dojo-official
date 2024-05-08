import asyncio
import functools
import json
import os
import random
import re
import sys
import textwrap
import time
import traceback
from typing import Any, Callable, List, Optional

import bittensor as bt
from regex import R
from unsync import unsync
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from strictjson import strict_json
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_fixed
from commons.llm.openai_proxy import Provider, get_openai_client
from commons.utils import PydanticUtils, log_retry_info
import os
import httpx
import asyncio

load_dotenv()


# class CodingQuestion(BaseModel):
#     question: str = Field(
#         description="Coding question to be solved by a software engineer"
#     )


# # Schema for the generated coding answer from LLM
# class FileObject(BaseModel):
#     filename: str = Field(description="Name of the file")
#     content: str = Field(description="Content of the file which can be code or json")
#     language: str = Field(description="Programming language of the file")


# class CodeAnswer(BaseModel):
#     files: List[FileObject] = Field(
#         description="Array of FileObject, that are part of the code solution. Must include index.html, and index.js a Javascript solution"
#     )
#     installation_commands: str = Field(
#         description="Terminal commands for the code to be able to run to install any third-party packages for the code to be able to run"
#     )
#     additional_notes: Optional[str] = Field(
#         description="Any additional notes or comments about the code solution"
#     )


# def build_code_generation_question_prompt(num_requirements: int) -> str:
#     bt.logging.info(f"Generating question with {num_requirements} requirements")
#     CODE_GEN_PROMPT = """
#     System:
#     - Generate a short, self-contained, challenging coding problem that requires the programmer to output an visualization from the piece of code with {num_requirements} requirements on the functionality of the interactions.
#     - The interactions must require the programmer to have a mental model of any objects being visualized.
#     - The question generated must require the programmer to code using only Python, or Javascript with HTML and CSS.
#     - You must not provide any example code snippets, because you must let the programmer solve the question by themselves.
#     - If the generated question is for Python, it must use built-in libraries. Strictly use mpld3 library functions for visualisation. Other python third-party libraries allowed are plotly, matplotlib and pandas==2.0.3.
#     - If the generated question is for Javascript, it should strictly command the usage of only built-in libraries or use visualization libraries like three.js, D3.js.

#     Coding Question:
#     """
#     return textwrap.dedent(CODE_GEN_PROMPT.format(num_requirements=num_requirements))


# def additional_notes_for_question_prompt(prompt: str) -> str:
#     ADDITIONAL_NOTES = """
#     Note:
#     - The visualization should be implemented in either Python using mpld3 with Plotly, Matplotlib, and Pandas (2.0.3) or in JavaScript with HTML and CSS using Three.js or D3.js.
#     - If mpld3 is used, ensure that mpld3.show() is used to display the plot.
#     """
#     return prompt + textwrap.dedent(ADDITIONAL_NOTES)


# def build_code_augmenter_prompt() -> str:
#     # TODO
#     pass


# def parse_openai_json_mode_response(completion_content: str):
#     pydantic_utils_keys = ["type", "schema"]
#     parsed = None
#     try:
#         json_content = extract_json(completion_content)
#         if json_content:
#             # successfully extracted json content
#             parsed = json.loads(json_content)
#         else:
#             parsed = json.loads(completion_content)
#     except json.JSONDecodeError as e:
#         bt.logging.info(f"Error occurred while parsing JSON response: {e}")
#     except Exception:
#         pass

#     if parsed:
#         for key in pydantic_utils_keys:
#             if key in parsed:
#                 parsed.pop(key)
#     return parsed


# def detect_chars_until_first_word(text: str):
#     pattern = r"^.*?(?=\b\w)"
#     match = re.search(pattern, text)
#     if not match:
#         return None
#     return match.group()


# def parse_code_response(result_object: CodeAnswer) -> CodeAnswer:
#     """Ensure that necessary files appended for python"""
#     result_object = append_codesandbox_files(result_object)
#     result_object = escape_double_quotes_in_files(result_object)
#     return result_object


# def escape_double_quotes_in_files(codeanswer_object: dict) -> CodeAnswer:
#     """Escapes double quotes in the content of each file in the CodeAnswer object."""
#     for file in codeanswer_object["files"]:
#         if "content" in file:
#             file["content"] = re.sub(r'(?<!\\)"', r"\"", file["content"])
#             file["content"] = file["content"].replace(r"\'", r"'")
#     return codeanswer_object


# def append_codesandbox_files(codeanswer_object: CodeAnswer) -> CodeAnswer:
#     """Appends necessary codesandbox files if any file's language key is Python."""
#     python_file_detected = False
#     requirements_file_exists = False
#     installation_commands = ""
#     python_file_name = "main.py"

#     files = codeanswer_object.get("files", [])
#     for file in files:
#         if isinstance(file, dict):
#             if file.get("language") == "python":
#                 python_file_detected = True
#                 python_file_name = file.get("filename", python_file_name)
#             if file.get("filename") == "requirements.txt":
#                 requirements_file_exists = True

#         if "installation_commands" in codeanswer_object:
#             installation_commands = codeanswer_object["installation_commands"]

#     bt.logging.info(f"requirements_file_exists: {requirements_file_exists}")
#     if python_file_detected:
#         devcontainer_file = {
#             "filename": ".devcontainer/devcontainer.json",
#             "content": json.dumps(
#                 {
#                     "name": "Devcontainer",
#                     "image": "mcr.microsoft.com/devcontainers/python:3.10",
#                     "customizations": {"vscode": {"extensions": ["ms-python.python"]}},
#                 },
#                 indent=2,
#             ),
#             "language": "json",
#         }

#         setup_task_command = (
#             "pip install -r requirements.txt"
#             if requirements_file_exists
#             else installation_commands
#         )
#         install_dependencies_command = (
#             "pip install -r requirements.txt"
#             if requirements_file_exists
#             else installation_commands
#         )
#         install_dependencies_files = (
#             ["requirements.txt"] if requirements_file_exists else []
#         )

#         codesandbox_tasks = CodeSandboxConfig(
#             setupTasks=(
#                 [
#                     Task(
#                         name="pip install -r requirements.txt",
#                         command=setup_task_command,
#                     )
#                 ]
#                 if setup_task_command
#                 else []
#             ),
#             tasks={
#                 "start": TaskDetail(
#                     name="start",
#                     command=f"python {python_file_name}",
#                     runAtStart=True,
#                     preview={"port": 8888},
#                     restartOn=RestartOn(files=[python_file_name]),
#                 ),
#                 "install-dependencies": TaskDetail(
#                     name="Installing Dependencies",
#                     command=install_dependencies_command,
#                     restartOn=RestartOn(files=install_dependencies_files),
#                 ),
#             },
#         )

#         codesandbox_tasks_json = codesandbox_tasks.json(indent=2, exclude_none=True)

#         codesandbox_tasks_file = {
#             "filename": ".codesandbox/tasks.json",
#             "content": codesandbox_tasks_json,
#             "language": "json",
#         }

#         if not requirements_file_exists:
#             requirements_file = {
#                 "filename": "requirements.txt",
#                 "content": "mpld3==0.5.10\npandas==2.0.3",
#                 "language": "text",
#             }

#         codeanswer_object["files"].extend(
#             [devcontainer_file, codesandbox_tasks_file, requirements_file]
#         )

#     return codeanswer_object


# def few_shot_example_outputs():
#     EXAMPLE_OUTPUTS = """
#     "question":"Write me a program that visualized our solar system, you may use python, javascript or pure HTML.",

#     Sample Answer Format:
#     {
#         "files": [
#             {
#             "filename": "index.js",
#             "content": "const canvas = document.getElementById(\"solarSystemCanvas\");\nconst ctx = canvas.getContext(\"2d\");\nconst infoPanel = document.getElementById(\"infoPanel\");\nconst speedSlider = document.getElementById(\"speedSlider\");\n\nconst planets = [\n  { name: \"Mercury\", orbitRadius: 50, orbitSpeed: 0.39, distanceFromSun: 39 },\n  { name: \"Venus\", orbitRadius: 100, orbitSpeed: 0.72, distanceFromSun: 72 },\n  { name: \"Earth\", orbitRadius: 150, orbitSpeed: 1, distanceFromSun: 100 },\n  { name: \"Mars\", orbitRadius: 200, orbitSpeed: 1.52, distanceFromSun: 152 },\n  {\n    name: \"Jupiter\",\n    orbitRadius: 300,\n    orbitSpeed: 11.86,\n    distanceFromSun: 520,\n  },\n  { name: \"Saturn\", orbitRadius: 400, orbitSpeed: 29.46, distanceFromSun: 958 },\n];\n\nlet currentTime = 0;\nlet simulationSpeed = 1;\n\nfunction drawPlanet(planet, angle) {\n  ctx.beginPath();\n  ctx.arc(\n    canvas.width / 2 + planet.orbitRadius * Math.cos(angle),\n    canvas.height / 2 + planet.orbitRadius * Math.sin(angle),\n    5,\n    0,\n    2 * Math.PI\n  );\n  ctx.fillStyle = \"blue\";\n  ctx.fill();\n  ctx.closePath();\n}\n\nfunction drawOrbit(planet) {\n  ctx.beginPath();\n  ctx.arc(\n    canvas.width / 2,\n    canvas.height / 2,\n    planet.orbitRadius,\n    0,\n    2 * Math.PI\n  );\n  ctx.strokeStyle = \"gray\";\n  ctx.stroke();\n  ctx.closePath();\n}\n\nfunction drawSun() {\n  ctx.beginPath();\n  ctx.arc(canvas.width / 2, canvas.height / 2, 10, 0, 2 * Math.PI);\n  ctx.fillStyle = \"yellow\";\n  ctx.fill();\n  ctx.closePath();\n}\n\nfunction updateInfoPanel(planet) {\n  infoPanel.innerHTML = `\n    <h2>${planet.name}</h2>\n    <p>Average Orbital Speed: ${planet.orbitSpeed} AU/year</p>\n    <p>Distance from Sun: ${planet.distanceFromSun} million km</p>\n  `;\n}\n\nfunction draw() {\n  ctx.clearRect(0, 0, canvas.width, canvas.height);\n  drawSun();\n\n  planets.forEach((planet, index) => {\n    const angle =\n      (currentTime * planet.orbitSpeed * simulationSpeed) % (2 * Math.PI);\n    drawOrbit(planet);\n    drawPlanet(planet, angle);\n\n    if (\n      ctx.isPointInPath(\n        canvas.width / 2,\n        canvas.height / 2 - planet.orbitRadius\n      )\n    ) {\n      updateInfoPanel(planet);\n    }\n  });\n\n  currentTime += 1 / 60;\n  requestAnimationFrame(draw);\n}\n\nspeedSlider.addEventListener(\"input\", (event) => {\n  simulationSpeed = event.target.value / 50;\n});\n\ndraw();\n",
#             "language": "javascript"
#             },
#             {
#             "filename": "index.html",
#             "content": "<!DOCTYPE html>\n<html>\n<head>\n<title>Page Title</title>\n</head>\n<body>\n<h1>Welcome</h1>\n<p>Hello world</p>\n<script src='index.js'></script>\n</body>\n</html>",
#             "language": "html"
#             }
#         ],
#         "installation_commands": "null",
#         "additional_notes": "The code uses built-in libraries so no additional commands are required."
#     }

#     "question": Create an interactive visualization of a cube in 3D space using Javascript with HTML and CSS. The visualization should meet the following requirements:
#     1. The cube should be rotatable in 3D space by clicking and dragging the mouse pointer.
#     2. The cube should change color when the mouse pointer is hovered over it.
#     3. The cube should maintain a consistent size regardless of the window size.
#     4. The cube can be rotated using arrow keys to move 90 degrees up, down, left, or right.
#     You should use libraries like three.js or D3.js to achieve this visualization. Please provide a self-contained HTML file containing the Javascript code and necessary HTML and CSS elements to visualize the cube.

#     Sample Answer Format:
#     {
#         "files": [
#             {
#             "filename": "index.html",
#             "content": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <meta charset=\"UTF-8\">\n    <meta name=\"viewport\" content=\"width=device-width,\n        initial-scale=1.0\">\n    <title>3D Cube Visualization</title>\n    <style>\n        body { margin: 0; }\n        canvas { display: block; }\n    </style>\n</head>\n<body>\n    <script src=\"https: //threejs.org/build/three.js\"></script>\n    <script>\n        // Setup scene, camera, and renderer\n        const scene = new THREE.Scene();\n        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\n        const renderer = new THREE.WebGLRenderer();\n        renderer.setSize(window.innerWidth, window.innerHeight);\n        document.body.appendChild(renderer.domElement);\n         // Create a cube\n        const geometry = new THREE.BoxGeometry();\n        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });\n        const cube = new THREE.Mesh(geometry, material);\n        scene.add(cube);\n         // Position the camera\n        camera.position.z = 5;\n         // Function to animate the scene\n        function animate() {\n            requestAnimationFrame(animate);\n            renderer.render(scene, camera);\n        }\n         // Mouse drag controls\n        let isDragging = false;\n        let previousMousePosition = { x: 0, y: 0 };\n         renderer.domElement.addEventListener('mousedown', (event) => {\n            isDragging = true;\n            previousMousePosition = { x: event.clientX, y: event.clientY };\n        });\n         renderer.domElement.addEventListener('mouseup', () => {\n            isDragging = false;\n        });\n         renderer.domElement.addEventListener('mousemove', (event) => {\n            if (isDragging) {\n                const deltaX = event.clientX - previousMousePosition.x;\n                const deltaY = event.clientY - previousMousePosition.y;\n                cube.rotation.y += deltaX * 0.01;\n                cube.rotation.x += deltaY * 0.01;\n                previousMousePosition = { x: event.clientX, y: event.clientY };\n            }\n        });\n         // Hover effect\n        renderer.domElement.addEventListener('mouseover', () => {\n            cube.material.color.set(0xff0000);\n        });\n         renderer.domElement.addEventListener('mouseout', () => {\n            cube.material.color.set(0x00ff00);\n        });\n         // Arrow key controls\n        document.addEventListener('keydown', (event) => {\n            switch (event.key) {\n                case 'ArrowUp':\n                    cube.rotation.x += Math.PI / 2;\n                    break;\n                case 'ArrowDown':\n                    cube.rotation.x -= Math.PI / 2;\n                    break;\n                case 'ArrowLeft':\n                    cube.rotation.y += Math.PI / 2;\n                    break;\n                case 'ArrowRight':\n                    cube.rotation.y -= Math.PI / 2;\n                    break;\n            }\n        });\n         // Start animation\n        animate();\n    </script>\n<script src=\"https: //threejs.org/build/three.js\"></script>\n</body>\n</html>>",
#             "language": "html"
#             }
#         ],
#         "installation_commands": "null",
#         "additional_notes": "include Three.js directly from a CDN by adding the following script tag to your HTML file: <script src=\"https://threejs.org/build/three.js\"></script>"
#     }

#     "question": Interactive Sine Wave Visualization
#     Write a Python program that generates an interactive visualization of a sine wave. The program should use matplotlib for plotting and mpld3 to convert the plot into an interactive HTML plot. The sine wave should span from 0 to 10 on the x-axis, with 100 points evenly distributed along this interval. The y-axis values should be the sine of the x-axis values. Your plot should have the title "Simple Plot" and appropriately labeled x and y-axes.
#     Ensure your solution:
#     - Imports necessary libraries
#     - Generates the data for the sine wave
#     - Creates the plot with titles and labels
#     - Converts the plot to an interactive HTML plot using mpld3

#     Sample Answer Format:
#     {
#         "files": [
#             {
#             "filename": "main.py",
#             "content": "import mpld3\r\nimport matplotlib.pyplot as plt\r\nimport numpy as np\r\n\r\n# Generate some data\r\nx = np.linspace(0, 10, 100)\r\ny = np.sin(x)\r\n\r\n# Create a plot\r\nplt.figure()\r\nplt.plot(x, y)\r\nplt.title('Simple Plot')\r\nplt.xlabel('x')\r\nplt.ylabel('y')\r\n\r\n# Convert the plot to an interactive HTML plot\r\n# html_plot = mpld3.fig_to_html(plt.gcf())\r\nmpld3.display()",
#             "language": "python"
#             },
#             {
#                 "filename": ".devcontainer/devcontainer.json",
#                 "content": "{\n  \"name\": \"Devcontainer\",\n  \"image\": \"mcr.microsoft.com/devcontainers/python:3.8-bookworm\",\n  \"customizations\": {\n    \"vscode\": {\n      \"extensions\": [\"ms-python.python\"]\n    }\n  }\n}",
#                 "language": "json"
#             },
#             {
#                 "filename": ".codesandbox/tasks.json",
#                 "content": "{\n  \/\/ These tasks will run in order when initializing your CodeSandbox project.\n  \"setupTasks\": [\n    {\n      \"name\": \"pip install -r requirements.txt\",\n      \"command\": \"pip install -r requirements.txt\"\n    }\n  ],\n\n  \/\/ These tasks can be run from CodeSandbox. Running one will open a log in the app.\n  \"tasks\": {\n    \"start\": {\n      \"name\": \"start\",\n      \"command\": \"python main.py\",\n      \"runAtStart\": true,\n      \"preview\": {\n        \"port\": 8050\n      },\n      \"restartOn\": {\n        \"files\": [\n          \"main.py\"\n        ],\n        \"branch\": false,\n        \"clone\": false,\n        \"resume\": false\n      }\n    },\n    \"install-dependencies\": {\n      \"name\": \"Installing Dependencies\",\n      \"command\": \"pip install -r requirements.txt\",\n      \"restartOn\": {\n        \"files\": [\n          \"requirements.txt\"\n        ],\n        \"branch\": false,\n        \"clone\": false,\n        \"resume\": false\n      }\n    }\n  }\n}",
#                 "language": "json"
#             },
#             {
#                 "filename": "requirements.txt",
#                 "content": "mpld3==0.5.10\npandas==2.0.3",
#                 "language": "text"
#             }
#         ],
#         "installation_commands": "pip install -r requirements.txt",
#         "additional_notes": "The code uses the dash library to visualise the data. The application is run using the main.py file. The CodeSandbox configuration is provided to run the application in a web-based environment. The requirements.txt file lists the dependencies required for the application."
#     }
#     """
#     return EXAMPLE_OUTPUTS


# def build_code_answer_prompt(question) -> str:
#     CODE_ANS_PROMPT = """
#     System:
#     - Your task is to solve the coding question below, according to the fields in the JSON_SCHEMA: {json_schema}.
#     - You must assume that you do not have access to the file system, therefore if any test data is provided, you must store it in memory appropriately in the necessary variable and not in a file.
#     - You must not provide any other text or explanations.
#     - You must provide all code required to ensure that your solution is complete.
#     - Do not leave out any details for brevity.
#     - Additionally, ensure that your code solution directly executes any functions required to provide the solution to the task.
#     - Your solution must not involve the useage of a terminal. If you require any inputs from the user, you must provide the functionality of the user input in your code.
#     - You are able to write to multiple output file foramts depending on your specific use case
#     - If your solution is in Python, ensure that the main file is named 'main.py'.
#     - If mpld3 is used, ensure that mpld3.show() is used to display the plot.
#     - Remember to include installation commands for any dependencies required for the code to run
#     - Ensure that a requirements.txt file is included if any third-party packages are required for the code to run.
#     - Ensure all output code is properly formatted with consistent quotation marks and special characters are correctly escaped to prevent syntax errors.
#     - The provided code solution should be directly executable without requiring modifications to run successfully.

#     Few-shot Example Outputs:
#     {few_shot_examples}

#     Question:
#     {question}

#     Answer according to the JSON_SCHEMA:
#     """

#     return textwrap.dedent(
#         CODE_ANS_PROMPT.format(
#             json_schema=PydanticUtils.build_response_format(CodeAnswer),
#             question=question,
#             few_shot_examples=few_shot_example_outputs(),
#         )
#     )


# def extract_strictjson_code(text) -> Optional[str]:
#     pattern = re.compile(r"```[a-zA-Z]+([\s\S]*?)```", re.MULTILINE)
#     matches = pattern.findall(text)
#     if not matches:
#         return None

#     parsed = "".join(matches)
#     return parsed.lstrip().rstrip()


# def extract_json(text) -> Optional[str]:
#     """Returns anything from between ```json ```"""
#     pattern = r"(?<=\`\`\`json\n)([\s\S]*?)(?=\n\`\`\`)"
#     extracted_code = re.search(pattern, text, re.DOTALL)
#     if extracted_code:
#         code_block = extracted_code.group(1)
#         return code_block
#     else:
#         print("No code block found between triple backticks.")
#         return None


# async def generate_question(client: AsyncOpenAI, model: str) -> Optional[str]:
#     MAX_RETRIES = 10
#     kwargs = {
#         "model": model,
#         "messages": [
#             {
#                 "role": "system",
#                 "content": build_code_generation_question_prompt(
#                     random.choices([3, 4, 5], weights=[0.5, 0.3, 0.2])[0]
#                 ),
#             }
#         ],
#         "temperature": 0.2,
#         "max_tokens": 8192,
#         "top_p": random.uniform(0.9, 1.0),
#     }

#     if model == "codellama/codellama-70b-instruct":
#         # perform provider routing
#         kwargs["extra_body"] = {
#             "provider": {"order": ["Together"]},
#         }

#     if model.startswith("openai"):
#         kwargs["seed"] = random.randint(0, 1e9)

#     try:
#         async for attempt in AsyncRetrying(
#             stop=stop_after_attempt(MAX_RETRIES),
#             wait=wait_fixed(0.10),
#             before_sleep=log_retry_info,
#         ):
#             with attempt:
#                 completion = await client.chat.completions.create(**kwargs)
#                 coding_question = completion.choices[0].message.content
#                 coding_question = additional_notes_for_question_prompt(coding_question)
#                 bt.logging.info(f"Generated question: {coding_question}")
#                 # attempt.retry_state.attempt_number
#                 return coding_question
#     except RetryError:
#         bt.logging.error(
#             f"Failed to generate completion after {MAX_RETRIES} attempts while generating question.",
#         )
#         pass

#     return None


# def on_error_update_kwargs(completion: ChatCompletion, kwargs_dict: dict):
#     if not hasattr(completion, "error"):
#         # false to tell caller kwargs weren't updated
#         return False, kwargs_dict

#     error_msg_json_str = completion.error.get("message") if completion.error else None
#     error_code = completion.error.get("code") if completion.error else None
#     # handle phind error
#     # data = """{'message': '{"error":{"message":"Phind/Phind-CodeLlama-34B-v2 is not supported for JSON mode/function calling","type":"invalid_request_error","param":null,"code":"constraints_model"}}', 'code': 400}"""
#     error_msg_json = {}
#     try:
#         if error_msg_json_str:
#             error_msg_json = json.loads(error_msg_json_str)
#             bt.logging.info(
#                 f"Got error code: {error_code} and error message: {error_msg_json}"
#             )
#             bt.logging.info("Successfully parsed json")
#     except json.JSONDecodeError:
#         pass
#     # handle no JSON mode
#     if (
#         error_msg_json
#         and "invalid_request_error" in error_msg_json_str["type"]
#         # and "not supported for JSON"
#         and error_code in [400, 422]
#     ):
#         kwargs_dict.pop("response_format")
#         bt.logging.warning("Updated kwargs due to JSON mode not supported...")

#     # kwargs were updated
#     return True, kwargs_dict


# def strictjson_llm_wrapper(system_prompt, user_prompt, model, args_dict):
#     """A wrapper for the AsyncOpenAI LLM call that strictjson will use.
#     Simply calls the unsync'd version of the async function and return the result.
#     """
#     return my_strict_json_llm_answer(
#         system_prompt, user_prompt, model, args_dict
#     ).result()


# @unsync
# async def my_strict_json_llm_answer(
#     system_prompt: str, user_prompt: str, model: str, kwargs: dict = {}
# ):
#     """Unsync'd version of an AsyncOpenAI call so we can call in inside of a synchronous context."""
#     async_client = get_openai_client(Provider.OPENROUTER)
#     result = await async_client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#         **kwargs,
#     )
#     return result.choices[0].message.content


# async def generate_strictjson_answer(sys, user, callable_llm: Callable):
#     loop = asyncio.get_running_loop()
#     # NOTE strict_json expects LLM call needs to be synchronous here
#     func = functools.partial(
#         strict_json,
#         system_prompt=sys,
#         user_prompt=user,
#         output_format=PydanticUtils.build_minimal_json(CodeAnswer),
#         llm=callable_llm,
#     )
#     result = await loop.run_in_executor(None, func)

#     return result


# async def generate_answer(model: str, question: str):
#     """Generates a coding question answer for a given coding question."""
#     MAX_RETRIES = 3

#     try:
#         async for attempt in AsyncRetrying(
#             stop=stop_after_attempt(MAX_RETRIES),
#             wait=wait_fixed(0.10),
#             before_sleep=log_retry_info,
#         ):
#             with attempt:
#                 # NOTE trying strictjson LLMs in order to stabilise the parsing of code outputs
#                 callable_llm = functools.partial(
#                     strictjson_llm_wrapper,
#                     model=model,
#                     args_dict={
#                         "temperature": 0.0,
#                         "max_tokens": 8192,
#                     },
#                 )
#                 completion = await generate_strictjson_answer(
#                     sys=build_code_answer_prompt(question),
#                     user="Remember to provide the code solution according your previous instructions.",
#                     callable_llm=callable_llm,
#                 )

#                 # TODO parse the response because of weird triple backticks or quotes
#                 # try:
#                 #     parsed = parse_code_response(completion)
#                 #     return model, parsed
#                 # except Exception as e:
#                 #     bt.logging.warning(
#                 #         "Failed to parse & extract code between triple backticks, naively returning original completion."
#                 #     )
#                 #     pass

#                 return model, completion
#     except RetryError:
#         bt.logging.error(
#             f"Failed to generate completion after {MAX_RETRIES} attempts for generating code answer for {model}"
#         )
#         pass
#     except Exception as e:
#         bt.logging.error(f"Error occurred while generating code answer: {e}")
#         pass

#     return model, None


# async def build_prompt_responses_pair():
#     import template

#     client = get_openai_client(Provider.OPENROUTER)
#     # use these models because we can specify seed
#     prompt = await generate_question(client, random.choice(template.GENERATOR_MODELS))

#     # NOTE @dev LLMs here were selected to be able to compare against the EvalPLus leaderboard
#     # randomly sampled from pool of models
#     answer_models = template.ANSWER_MODELS
#     IS_TEST = os.getenv("IS_TEST", False)
#     num_samples = len(template.ANSWER_MODELS) if IS_TEST else 4
#     sel_ans_models = random.sample(answer_models, num_samples)

#     results = await asyncio.gather(
#         *[generate_answer(ans_model, prompt) for ans_model in sel_ans_models]
#     )

#     res = {"prompt": prompt, "responses": []}
#     for model, result in results:
#         if not result:
#             continue
#         result = parse_code_response(result)
#         res["responses"].append(
#             {
#                 "model": model,
#                 "completion": {
#                     "files": result["files"],
#                     "installation_commands": result["installation_commands"],
#                     "additional_notes": result["additional_notes"],
#                 },
#             }
#         )
#     return res


# async def main():
#     res = await build_prompt_responses_pair()
#     bt.logging.info(f"{res=}")


# if __name__ == "__main__":
#     asyncio.run(main())


class FileObject(BaseModel):
    filename: str = Field(description="Name of the file")
    content: str = Field(description="Content of the file which can be code or json")
    language: str = Field(description="Programming language of the file")


class CodeAnswer(BaseModel):
    files: List[FileObject] = Field(description="List of FileObjects")
    installation_commands: str = Field(
        description="Terminal commands for the code to be able to run to install any third-party packages for the code to be able to run"
    )
    additional_notes: Optional[str] = Field(
        description="Any additional notes or comments about the code solution"
    )


class Response(BaseModel):
    model: str
    completion: CodeAnswer


class SyntheticQA(BaseModel):
    prompt: str
    responses: List[Response]


SYNTHETIC_API_BASE_URL = os.getenv("SYNTHETIC_API_URL")


class SyntheticAPI:
    _queue = asyncio.Queue()
    _lock = asyncio.Lock()
    _queue_size = 10

    @classmethod
    async def populate_queue(cls):
        while True:
            print(f"Checking queue size...{cls._queue.qsize()}")
            if cls._queue.qsize() >= cls._queue_size:
                continue
            num_items = cls._queue_size - cls._queue.qsize()
            tasks = [cls._generate_synthetic_qa() for _ in range(num_items)]
            # Access task results and handle them appropriately
            completed_tasks = await asyncio.gather(*tasks)
            for task_result in completed_tasks:
                if not task_result:
                    continue
                async with cls._lock:
                    await cls._queue.put(task_result)
            await asyncio.sleep(1)

    @classmethod
    async def _generate_synthetic_qa(cls):
        # TODO @dev improve this speed
        path = f"{SYNTHETIC_API_BASE_URL}/api/synthetic-gen"
        async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
            response = await client.get(path)
            response_json = response.json()
            try:
                synthetic_qa = SyntheticQA.parse_obj(response_json["body"])
                return synthetic_qa
            except Exception as exc:
                traceback.print_exc()

    @classmethod
    async def get_qa(cls):
        if cls._queue.qsize() == 0:
            return await cls._generate_synthetic_qa()

        async with cls._lock:
            return await cls._queue.get()


if __name__ == "__main__":

    async def simulate_vali():
        while True:
            start_time = time.perf_counter()
            synthetic_qa = await SyntheticAPI.get_qa()
            elapsed_time = time.perf_counter() - start_time
            print(f"Time taken to get QA: {elapsed_time:.2f} seconds")
            print(synthetic_qa)

    async def main():
        await asyncio.gather(*[simulate_vali(), SyntheticAPI.populate_queue()])

    asyncio.run(main())
