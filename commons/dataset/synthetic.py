import asyncio
import functools
import json
import os
import random
import re
import sys
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

sys.path.append("./")
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


# Schema for the generated coding answer from LLM
class FileObject(BaseModel):
    filename: str = Field(description="Name of the file")
    content: str = Field(description="Content of the file")
    language: str = Field(description="Programming language of the file")


class CodeAnswer(BaseModel):
    files: List[FileObject] = Field(description="Code solution to the question")
    installation_commands: str = Field(
        description="Terminal commands for the code to be able to run to install any third-party packages for the code to be able to run"
    )
    additional_notes: Optional[str] = Field(
        description="Any additional notes or comments about the code solution"
    )


# Schema for tasks.json file needed for CodeSandbox
class Task(BaseModel):
    name: str
    command: str


class RestartOn(BaseModel):
    files: List[str] = []
    branch: bool = False
    clone: bool = False
    resume: bool = False


class TaskDetail(BaseModel):
    name: str
    command: str
    runAtStart: Optional[bool] = None
    preview: Optional[dict[str, int]] = None
    restartOn: RestartOn = RestartOn()


class CodeSandboxConfig(BaseModel):
    setupTasks: List[Task] = Field(
        description="These tasks will run in order when initializing your CodeSandbox project."
    )
    tasks: dict[str, TaskDetail] = Field(
        description="These tasks can be run from CodeSandbox. Running one will open a log in the app."
    )


def build_code_generation_question_prompt(num_requirements: int) -> str:
    bt.logging.info(f"Generating question with {num_requirements} requirements")
    CODE_GEN_PROMPT = """
    System:
    - Generate a short, self-contained, challenging coding problem that requires the programmer to output an visualization from the piece of code with {num_requirements} requirements on the functionality of the interactions.
    - The interactions must require the programmer to have a mental model of any objects being visualized.
    - The question generated must require the programmer to code using only Python, or Javascript with HTML and CSS.
    - You must not provide any example code snippets, because you must let the programmer solve the question by themselves.
    - If the generated question is for Python, it must use built-in libraries. Strictly use dash==2.16.1 library functions for visualisation. Other python third-party libraries allowed are plotly, matplotlib and pandas==2.0.3.
    - If the generated question is for Javascript, it should strictly command the usage of only built-in libraries or use visualization libraries like three.js, D3.js.

    Coding Question:
    """
    return textwrap.dedent(CODE_GEN_PROMPT.format(num_requirements=num_requirements))


def additional_notes_for_question_prompt(prompt: str) -> str:
    ADDITIONAL_NOTES = """
    Note:
    - The visualization should be implemented in either Python using Dash (2.16.1) with Plotly, Matplotlib, and Pandas (2.0.3) or in JavaScript with HTML and CSS using Three.js or D3.js.
    """
    return prompt + textwrap.dedent(ADDITIONAL_NOTES)


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
                if len(code_text):
                    strictjson_response["code"] = code_text
    except Exception as e:
        pass

    return strictjson_response


def augment_python_files(strictjson_response: dict[str, Any]) -> dict:
    """Appends specific files if any file's language key is Python."""
    python_file_detected = False
    requirements_file_exists = False
    installation_commands = ""
    python_file_name = "main.py"

    for file in strictjson_response.get("files", []):
        if file.get("language") == "python":
            python_file_detected = True
            python_file_name = file.get("filename", python_file_name)
        if file.get("filename") == "requirements.txt":
            requirements_file_exists = True
        if "installation_commands" in strictjson_response:
            installation_commands = strictjson_response["installation_commands"]

    if python_file_detected:
        devcontainer_file = {
            "filename": ".devcontainer/devcontainer.json",
            "content": json.dumps(
                {
                    "name": "Devcontainer",
                    "image": "mcr.microsoft.com/devcontainers/python:3.8-bookworm",
                    "customizations": {"vscode": {"extensions": ["ms-python.python"]}},
                },
                indent=2,
            ),
            "language": "json",
        }

        setup_task_command = (
            "pip install -r requirements.txt"
            if requirements_file_exists
            else installation_commands
        )
        install_dependencies_command = (
            "pip install -r requirements.txt"
            if requirements_file_exists
            else installation_commands
        )
        install_dependencies_files = (
            ["requirements.txt"] if requirements_file_exists else []
        )

        codesandbox_tasks = CodeSandboxConfig(
            setupTasks=(
                [
                    Task(
                        name="pip install -r requirements.txt",
                        command=setup_task_command,
                    )
                ]
                if setup_task_command
                else []
            ),
            tasks={
                "start": TaskDetail(
                    name="start",
                    command=f"python {python_file_name}",
                    runAtStart=True,
                    preview={"port": 8050},
                    restartOn=RestartOn(files=[python_file_name]),
                ),
                "install-dependencies": TaskDetail(
                    name="Installing Dependencies",
                    command=install_dependencies_command,
                    restartOn=RestartOn(files=install_dependencies_files),
                ),
            },
        )

        codesandbox_tasks_json = codesandbox_tasks.json(indent=2)
        # Serialize again to escape the string for embedding
        escaped_codesandbox_tasks_json = json.dumps(codesandbox_tasks_json)

        codesandbox_tasks_file = {
            "filename": ".codesandbox/tasks.json",
            "content": escaped_codesandbox_tasks_json,
            "language": "json",
        }

        strictjson_response["files"].extend([devcontainer_file, codesandbox_tasks_file])

    return strictjson_response


def few_shot_example_outputs():
    EXAMPLE_OUTPUTS = """
    "question":"Write me a program that visualized our solar system, you may use python, javascript or pure HTML.",

    Sample Answer Format:
        {
            "files": [
                {
                "filename": "index.js",
                "content": "const canvas = document.getElementById(\"solarSystemCanvas\");\nconst ctx = canvas.getContext(\"2d\");\nconst infoPanel = document.getElementById(\"infoPanel\");\nconst speedSlider = document.getElementById(\"speedSlider\");\n\nconst planets = [\n  { name: \"Mercury\", orbitRadius: 50, orbitSpeed: 0.39, distanceFromSun: 39 },\n  { name: \"Venus\", orbitRadius: 100, orbitSpeed: 0.72, distanceFromSun: 72 },\n  { name: \"Earth\", orbitRadius: 150, orbitSpeed: 1, distanceFromSun: 100 },\n  { name: \"Mars\", orbitRadius: 200, orbitSpeed: 1.52, distanceFromSun: 152 },\n  {\n    name: \"Jupiter\",\n    orbitRadius: 300,\n    orbitSpeed: 11.86,\n    distanceFromSun: 520,\n  },\n  { name: \"Saturn\", orbitRadius: 400, orbitSpeed: 29.46, distanceFromSun: 958 },\n];\n\nlet currentTime = 0;\nlet simulationSpeed = 1;\n\nfunction drawPlanet(planet, angle) {\n  ctx.beginPath();\n  ctx.arc(\n    canvas.width / 2 + planet.orbitRadius * Math.cos(angle),\n    canvas.height / 2 + planet.orbitRadius * Math.sin(angle),\n    5,\n    0,\n    2 * Math.PI\n  );\n  ctx.fillStyle = \"blue\";\n  ctx.fill();\n  ctx.closePath();\n}\n\nfunction drawOrbit(planet) {\n  ctx.beginPath();\n  ctx.arc(\n    canvas.width / 2,\n    canvas.height / 2,\n    planet.orbitRadius,\n    0,\n    2 * Math.PI\n  );\n  ctx.strokeStyle = \"gray\";\n  ctx.stroke();\n  ctx.closePath();\n}\n\nfunction drawSun() {\n  ctx.beginPath();\n  ctx.arc(canvas.width / 2, canvas.height / 2, 10, 0, 2 * Math.PI);\n  ctx.fillStyle = \"yellow\";\n  ctx.fill();\n  ctx.closePath();\n}\n\nfunction updateInfoPanel(planet) {\n  infoPanel.innerHTML = `\n    <h2>${planet.name}</h2>\n    <p>Average Orbital Speed: ${planet.orbitSpeed} AU/year</p>\n    <p>Distance from Sun: ${planet.distanceFromSun} million km</p>\n  `;\n}\n\nfunction draw() {\n  ctx.clearRect(0, 0, canvas.width, canvas.height);\n  drawSun();\n\n  planets.forEach((planet, index) => {\n    const angle =\n      (currentTime * planet.orbitSpeed * simulationSpeed) % (2 * Math.PI);\n    drawOrbit(planet);\n    drawPlanet(planet, angle);\n\n    if (\n      ctx.isPointInPath(\n        canvas.width / 2,\n        canvas.height / 2 - planet.orbitRadius\n      )\n    ) {\n      updateInfoPanel(planet);\n    }\n  });\n\n  currentTime += 1 / 60;\n  requestAnimationFrame(draw);\n}\n\nspeedSlider.addEventListener(\"input\", (event) => {\n  simulationSpeed = event.target.value / 50;\n});\n\ndraw();\n",
                "language": "javascript"
                },
                {
                "filename": "index.html",
                "content": "<!DOCTYPE html>\n<html>\n<head>\n<title>Page Title</title>\n</head>\n<body>\n<h1>Welcome</h1>\n<p>Hello world</p>\n<script src='index.js'></script>\n</body>\n</html>",
                "language": "html"
                }
            ],
            "installation_commands": "null",
            "additional_notes": "The code uses built-in libraries so no additional commands are required."
            }
        },

    "question": Create an interactive visualization of a cube in 3D space using Javascript with HTML and CSS. The visualization should meet the following requirements:
    1. The cube should be rotatable in 3D space by clicking and dragging the mouse pointer.
    2. The cube should change color when the mouse pointer is hovered over it.
    3. The cube should maintain a consistent size regardless of the window size.
    4. The cube can be rotated using arrow keys to move 90 degrees up, down, left, or right.
    You should use libraries like three.js or D3.js to achieve this visualization. Please provide a self-contained HTML file containing the Javascript code and necessary HTML and CSS elements to visualize the cube.

    Sample Answer Format:
    {
    "files": [
        {
        "filename": "index.html",
        "content": "<!DOCTYPE html>\\n<html lang=\"en\">\\n<head>\\n    <meta charset=\"UTF-8\">\\n    <meta name=\"viewport\" content=\"width=device-width,\\n        initial-scale=1.0\">\\n    <title>3D Cube Visualization</title>\\n    <style>\\n        body { margin: 0; }\\n        canvas { display: block; }\\n    </style>\\n</head>\\n<body>\\n    <script src=\"https: //threejs.org/build/three.js\"></script>\\n    <script>\\n        // Setup scene, camera, and renderer\\n        const scene = new THREE.Scene();\\n        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);\\n        const renderer = new THREE.WebGLRenderer();\\n        renderer.setSize(window.innerWidth, window.innerHeight);\\n        document.body.appendChild(renderer.domElement);\\n         // Create a cube\\n        const geometry = new THREE.BoxGeometry();\\n        const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });\\n        const cube = new THREE.Mesh(geometry, material);\\n        scene.add(cube);\\n         // Position the camera\\n        camera.position.z = 5;\\n         // Function to animate the scene\\n        function animate() {\\n            requestAnimationFrame(animate);\\n            renderer.render(scene, camera);\\n        }\\n         // Mouse drag controls\\n        let isDragging = false;\\n        let previousMousePosition = { x: 0, y: 0 };\\n         renderer.domElement.addEventListener('mousedown', (event) => {\\n            isDragging = true;\\n            previousMousePosition = { x: event.clientX, y: event.clientY };\\n        });\\n         renderer.domElement.addEventListener('mouseup', () => {\\n            isDragging = false;\\n        });\\n         renderer.domElement.addEventListener('mousemove', (event) => {\\n            if (isDragging) {\\n                const deltaX = event.clientX - previousMousePosition.x;\\n                const deltaY = event.clientY - previousMousePosition.y;\\n                cube.rotation.y += deltaX * 0.01;\\n                cube.rotation.x += deltaY * 0.01;\\n                previousMousePosition = { x: event.clientX, y: event.clientY };\\n            }\\n        });\\n         // Hover effect\\n        renderer.domElement.addEventListener('mouseover', () => {\\n            cube.material.color.set(0xff0000);\\n        });\\n         renderer.domElement.addEventListener('mouseout', () => {\\n            cube.material.color.set(0x00ff00);\\n        });\\n         // Arrow key controls\\n        document.addEventListener('keydown', (event) => {\\n            switch (event.key) {\\n                case 'ArrowUp':\\n                    cube.rotation.x += Math.PI / 2;\\n                    break;\\n                case 'ArrowDown':\\n                    cube.rotation.x -= Math.PI / 2;\\n                    break;\\n                case 'ArrowLeft':\\n                    cube.rotation.y += Math.PI / 2;\\n                    break;\\n                case 'ArrowRight':\\n                    cube.rotation.y -= Math.PI / 2;\\n                    break;\\n            }\\n        });\\n         // Start animation\\n        animate();\\n    </script>\\n<script src=\"https: //threejs.org/build/three.js\"></script>\\n</body>\\n</html>>",
        "language": "html"
        }
    ],
    "installation_commands": "null",
    "additional_notes": "include Three.js directly from a CDN by adding the following script tag to your HTML file:<script src="https: //threejs.org/build/three.js"></script>"
    }
    },

    "question": Interactive Data Visualization with Dash and Plotly
    You are tasked with creating an interactive web application using Dash and Plotly to visualize a dataset. The dataset consists of points categorized by a type of fruit, each with an x and y coordinate and an associated custom data value.
    Your application should display a scatter plot of the dataset, with points colored according to their fruit category. It should also provide interactive features to display data related to user interactions with the plot, such as hovering over points, clicking on points, selecting multiple points using lasso or rectangle tools, and zooming or panning the plot.
    Requirements:
    1. Come up with a dataset to create a scatter plot. The dataset should the following columns: x, y, customdata, and fruit.
    2. Color the points in the scatter plot based on the fruit column.
    3. Implement interactive features that display:
    - The data of the point(s) hovered over.
    - The data of the point(s) clicked on.
    - The data of the point(s) selected using lasso or rectangle selection tools.
    - Data related to zooming or panning actions on the plot.
    Additional Information:
    - You should use Dash for the web application framework and Plotly for creating the scatter plot.
    - The application layout should be organized, with the plot and interactive data sections arranged for easy viewing.
    - Ensure that your application is responsive to different types of user interactions with the plot, such as hovering, clicking, selecting, and zooming/panning.

    Sample Answer Format:
    {
    "files": [
        {
        "filename": "main.py",
        "content": "from dash import Dash, dcc, html, Input, Output, callback\n\nimport plotly.express as px\n\nimport json\nimport pandas as pd\n\nexternal_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']\n\napp = Dash(__name__, external_stylesheets=external_stylesheets)\n\nstyles = {\n    'pre': {\n        'border': 'thin lightgrey solid',\n        'overflowX': 'scroll'\n    }\n}\n\ndf = pd.DataFrame({\n    \"x\": [1,2,1,2],\n    \"y\": [1,2,3,4],\n    \"customdata\": [1,2,3,4],\n    \"fruit\": [\"apple\", \"apple\", \"orange\", \"orange\"]\n})\n\nfig = px.scatter(df, x=\"x\", y=\"y\", color=\"fruit\", custom_data=[\"customdata\"])\n\nfig.update_layout(clickmode='event+select')\n\nfig.update_traces(marker_size=20)\n\napp.layout = html.Div([\n    dcc.Graph(\n        id='basic-interactions',\n        figure=fig\n    ),\n\n    html.Div(className='row', children=[\n        html.Div([\n            dcc.Markdown(\"\"\"\n                **Hover Data**\n\n                Mouse over values in the graph.\n            \"\"\"),\n            html.Pre(id='hover-data', style=styles['pre'])\n        ], className='three columns'),\n\n        html.Div([\n            dcc.Markdown(\"\"\"\n                **Click Data**\n\n                Click on points in the graph.\n            \"\"\"),\n            html.Pre(id='click-data', style=styles['pre']),\n        ], className='three columns'),\n\n        html.Div([\n            dcc.Markdown(\"\"\"\n                **Selection Data**\n\n                Choose the lasso or rectangle tool in the graph's menu\n                bar and then select points in the graph.\n\n                Note that if `layout.clickmode = 'event+select'`, selection data also\n                accumulates (or un-accumulates) selected data if you hold down the shift\n                button while clicking.\n            \"\"\"),\n            html.Pre(id='selected-data', style=styles['pre']),\n        ], className='three columns'),\n\n        html.Div([\n            dcc.Markdown(\"\"\"\n                **Zoom and Relayout Data**\n\n                Click and drag on the graph to zoom or click on the zoom\n                buttons in the graph's menu bar.\n                Clicking on legend items will also fire\n                this event.\n            \"\"\"),\n            html.Pre(id='relayout-data', style=styles['pre']),\n        ], className='three columns')\n    ])\n])\n\n\n@callback(\n    Output('hover-data', 'children'),\n    Input('basic-interactions', 'hoverData'))\ndef display_hover_data(hoverData):\n    return json.dumps(hoverData, indent=2)\n\n\n@callback(\n    Output('click-data', 'children'),\n    Input('basic-interactions', 'clickData'))\ndef display_click_data(clickData):\n    return json.dumps(clickData, indent=2)\n\n\n@callback(\n    Output('selected-data', 'children'),\n    Input('basic-interactions', 'selectedData'))\ndef display_selected_data(selectedData):\n    return json.dumps(selectedData, indent=2)\n\n\n@callback(\n    Output('relayout-data', 'children'),\n    Input('basic-interactions', 'relayoutData'))\ndef display_relayout_data(relayoutData):\n    return json.dumps(relayoutData, indent=2)\n\n\nif __name__ == '__main__':\n    app.run(debug=True)\n",
        "language": "python"
        },
        {
            "filename": ".devcontainer/devcontainer.json",
            "content": "{\n  \"name\": \"Devcontainer\",\n  \"image\": \"mcr.microsoft.com/devcontainers/python:3.8-bookworm\",\n  \"customizations\": {\n    \"vscode\": {\n      \"extensions\": [\"ms-python.python\"]\n    }\n  }\n}",
            "language": "json"
        },
        {
            "filename": ".codesandbox/tasks.json",
            "content": "{\n  \/\/ These tasks will run in order when initializing your CodeSandbox project.\n  \"setupTasks\": [\n    {\n      \"name\": \"pip install -r requirements.txt\",\n      \"command\": \"pip install -r requirements.txt\"\n    }\n  ],\n\n  \/\/ These tasks can be run from CodeSandbox. Running one will open a log in the app.\n  \"tasks\": {\n    \"start\": {\n      \"name\": \"start\",\n      \"command\": \"python main.py\",\n      \"runAtStart\": true,\n      \"preview\": {\n        \"port\": 8050\n      },\n      \"restartOn\": {\n        \"files\": [\n          \"main.py\"\n        ],\n        \"branch\": false,\n        \"clone\": false,\n        \"resume\": false\n      }\n    },\n    \"install-dependencies\": {\n      \"name\": \"Installing Dependencies\",\n      \"command\": \"pip install -r requirements.txt\",\n      \"restartOn\": {\n        \"files\": [\n          \"requirements.txt\"\n        ],\n        \"branch\": false,\n        \"clone\": false,\n        \"resume\": false\n      }\n    }\n  }\n}",
            "language": "json"
        },
        {
            "filename": "requirements.txt",
            "content": "dash==2.16.1\npandas==2.0.3",
            "language": "text"
        }
    ],
    "installation_commands": "pip install -r requirements.txt",
    "additional_notes": "The code uses the dash library to visualise the data. The application is run using the main.py file. The CodeSandbox configuration is provided to run the application in a web-based environment. The requirements.txt file lists the dependencies required for the application."
    }
    },

    """
    return EXAMPLE_OUTPUTS


def build_code_answer_prompt(question) -> str:
    CODE_ANS_PROMPT = """
    System:
    - Your task is to solve the coding question below, according to the fields in the JSON_SCHEMA: {json_schema}.
    - You must assume that you do not have access to the file system, therefore if any test data is provided, you must store it in memory appropriately in the necessary variable and not in a file.
    - You must not provide any other text or explanations.
    - You must provide all code required to ensure that your solution is complete.
    - Do not leave out any details for brevity.
    - Additionally, ensure that your code solution directly executes any functions required to provide the solution to the task.
    - Your solution must not involve the useage of a terminal. If you require any inputs from the user, you must provide the functionality of the user input in your code.
    - You are able to write to multiple output file foramts depending on your specific use case
    - If your solution is in Python, ensure that the main file is named 'main.py'.
    - Remember to include installation commands for any dependencies required for the code to run
    - Ensure all output code is properly formatted with consistent quotation marks and special characters are correctly escaped to prevent syntax errors.
    - The provided code solution should be directly executable without requiring modifications to run successfully.

    Few-shot Example Outputs:
    {few_shot_examples}
    
    Question:
    {question}

    Answer according to the JSON_SCHEMA:
    """

    return textwrap.dedent(
        CODE_ANS_PROMPT.format(
            json_schema=PydanticUtils.build_response_format(CodeAnswer),
            question=question,
            few_shot_examples=few_shot_example_outputs(),
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
                "content": build_code_generation_question_prompt(
                    random.choices([3, 4, 5], weights=[0.5, 0.3, 0.2])[0]
                ),
            }
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
        "top_p": random.uniform(0.9, 1.0),
    }

    if model == "codellama/codellama-70b-instruct":
        # perform provider routing
        kwargs["extra_body"] = {
            "provider": {"order": ["Together"]},
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
                coding_question = additional_notes_for_question_prompt(coding_question)
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


async def generate_answer(model: str, question: str):
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
                completion = augment_python_files(completion)

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
    except Exception as e:
        bt.logging.error(f"Error occurred while generating code answer: {e}")
        pass

    return model, None


async def build_prompt_responses_pair():
    import template

    client = get_openai_client(Provider.OPENROUTER)
    # use these models because we can specify seed
    prompt = await generate_question(client, random.choice(template.GENERATOR_MODELS))

    # NOTE @dev LLMs here were selected to be able to compare against the EvalPLus leaderboard
    # randomly sampled from pool of models
    answer_models = template.ANSWER_MODELS
    IS_TEST = os.getenv("IS_TEST", False)
    num_samples = len(template.answer_models) if IS_TEST else 4
    sel_ans_models = random.sample(answer_models, num_samples)

    results = await asyncio.gather(
        *[generate_answer(ans_model, prompt) for ans_model in sel_ans_models]
    )
    res = {"prompt": prompt, "responses": []}
    for model, result in results:
        if not result:
            continue
        res["responses"].append(
            {
                "model": model,
                "completion": {
                    "files": result["files"],
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
