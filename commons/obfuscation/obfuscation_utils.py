import argparse
import base64
import hashlib
import os
import random
import re
import string
import subprocess
import tempfile
from typing import Callable

from bittensor.btlogging import logging as logger
from bs4 import BeautifulSoup


# Obfuscator base class
class Obfuscator:
    @staticmethod
    def generate_random_string(length=8):
        return "".join(
            random.choices(string.ascii_letters, k=1)
            + random.choices(string.ascii_letters + string.digits, k=length - 1)
        )

    @staticmethod
    def simple_encrypt(text, key):
        return base64.b64encode(bytes([ord(c) ^ key for c in text])).decode()

    @classmethod
    def obfuscate(cls, content):
        raise NotImplementedError("Subclasses must implement this method")


# Obfuscator for HTML content
# Encrypts the HTML content and generates a JavaScript snippet to decrypt it
# The JavaScript snippet is then embedded in the HTML content
class HTMLObfuscator(Obfuscator):
    @staticmethod
    def generate_random_string(length=8):
        return "".join(
            random.choices(string.ascii_letters, k=1)
            + random.choices(string.ascii_letters + string.digits, k=length - 1)
        )

    @staticmethod
    def simple_encrypt(text, key):
        return base64.b64encode(bytes(c ^ key for c in text.encode())).decode()

    @classmethod
    def obfuscate(cls, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        scripts = soup.find_all("script")

        # Obfuscate the remaining HTML
        body_content = str(soup.body)
        body_content = re.sub(r"\s+", " ", body_content).replace("> <", "><")

        encryption_key = random.randint(1, 255)
        encrypted_content = cls.simple_encrypt(body_content, encryption_key)

        decrypt_func, result_var = (
            cls.generate_random_string(),
            cls.generate_random_string(),
        )

        js_code = (
            f"function {decrypt_func}(e,t){{var r=atob(e),n='';for(var i=0;i<r.length;i++){{n+=String.fromCharCode(r.charCodeAt(i)^t)}}return n}}"
            f"var {result_var}={decrypt_func}('{encrypted_content}',{encryption_key});"
            f"document.body.innerHTML={result_var};"
        )

        new_script = soup.new_tag("script")
        new_script.string = js_code

        soup.body.clear()
        soup.body.append(new_script)
        soup.body.extend(scripts)

        return str(soup)


# Obfuscator for JavaScript code
# Uses UglifyJS to minify and obfuscate the JavaScript code
class JSObfuscator(Obfuscator):
    UGLIFYJS_COMMAND = [
        "uglifyjs",
        "--compress",
        "--mangle",
        "--mangle-props",
        "--toplevel",
    ]

    @staticmethod
    def obfuscate(js_code):
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".js", delete=True
        ) as temp_file:
            temp_file.write(js_code)
            temp_file.flush()

            try:
                result = subprocess.run(
                    JSObfuscator.UGLIFYJS_COMMAND + [temp_file.name],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                logger.error(f"Error occurred while obfuscating: {e}")
                return js_code

    @classmethod
    def obfuscate_html(cls, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        for script in soup.find_all("script", string=True):
            script.string = cls.obfuscate(script.string)
        return str(soup)


def obfuscate_html_and_js(html_content):
    return HTMLObfuscator.obfuscate(JSObfuscator.obfuscate_html(html_content))


def process_file(input_file: str, output_file: str, obfuscation_func: Callable):
    try:
        with open(input_file, encoding="utf-8") as file:
            original_content = file.read()
    except FileNotFoundError:
        logger.error(f"Error: The file '{input_file}' was not found.")
        return
    except OSError:
        logger.error(f"Error: Could not read the file '{input_file}'.")
        return

    obfuscated = obfuscation_func(original_content)

    try:
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(obfuscated)
        logger.info(f"Obfuscated content has been written to '{output_file}'")

        # Calculate and display hashes to show difference
        original_hash = hashlib.md5(original_content.encode()).hexdigest()
        obfuscated_hash = hashlib.md5(obfuscated.encode()).hexdigest()
        logger.info(f"\nOriginal content MD5: {original_hash}")
        logger.info(f"Obfuscated content MD5: {obfuscated_hash}")
    except OSError:
        logger.error(f"Error: Could not write to the file '{output_file}'.")


# Function to test the obfuscation
# Command to run: python obfuscation_utils.py input.html
def main():
    parser = argparse.ArgumentParser(
        description="Obfuscate HTML and JavaScript content"
    )
    parser.add_argument("input_file", help="Path to the input HTML file")
    parser.add_argument(
        "-o", "--output", help="Path to the output obfuscated HTML file (optional)"
    )
    parser.add_argument(
        "--mode",
        choices=["html", "js", "both"],
        default="both",
        help="Obfuscation mode",
    )
    args = parser.parse_args()

    # Generate default output filename based on input filename
    input_filename = os.path.basename(args.input_file)
    input_name, input_ext = os.path.splitext(input_filename)
    output_file = args.output or f"{input_name}_obfuscated{input_ext}"

    if args.mode == "html":
        obfuscation_func = HTMLObfuscator.obfuscate
    elif args.mode == "js":
        obfuscation_func = JSObfuscator.obfuscate_html
    else:
        obfuscation_func = obfuscate_html_and_js

    process_file(args.input_file, output_file, obfuscation_func)


if __name__ == "__main__":
    main()
