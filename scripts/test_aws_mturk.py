import os
from commons.objects import ObjectManager

from commons.human_feedback.aws_mturk import MTurkUtils
from template.protocol import Completion
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    factory = ObjectManager()
    config = factory.get_config()
    MTurkUtils.create_mturk_task(
        prompt="This is a test prompt",
        completions=[
            Completion(text="My name is Alice!"),
            Completion(text="My name is Bob!"),
        ],
        title="Prompt & Completion Task",
    )
