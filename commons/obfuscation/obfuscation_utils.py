import argparse
import base64
import hashlib
import os
import random
import re
import string
import subprocess
import tempfile
import time
from typing import Callable

from bittensor.btlogging import logging as logger
from bs4 import BeautifulSoup
from jsmin import jsmin


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
        error_var, temp_var = cls.generate_random_string(), cls.generate_random_string()

        # Add more randomization to the JavaScript code
        js_operations = [
            f"var {temp_var}=atob(e);",
            f"var {result_var}='';",
            f"for(var {cls.generate_random_string()}=0;{cls.generate_random_string()}<{temp_var}.length;{cls.generate_random_string()}++){{",
            f"{result_var}+=String.fromCharCode({temp_var}.charCodeAt({cls.generate_random_string()})^t);",
            "}",
        ]
        random.shuffle(js_operations)

        js_code = (
            f"function {decrypt_func}(e,t){{try{{{';'.join(js_operations)}return {result_var}}}catch({error_var}){{console.error('Decryption failed:',{error_var});return e}}}}"
            f"var {result_var}={decrypt_func}('{encrypted_content}',{encryption_key});"
            f"if({result_var}.indexOf('<')!==-1){{document.body.innerHTML={result_var};}}else{{console.error('Decryption produced invalid HTML');document.body.innerHTML=atob('{encrypted_content}');}}"
        )

        if random.choice([True, False]):
            dummy_func = cls.generate_random_string()
            js_code = f"function {dummy_func}(){{return Math.random()<0.5}}" + js_code

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
    MAX_RETRIES = 5
    RETRY_DELAY = 1  # seconds

    @staticmethod
    def is_uglifyjs_available():
        try:
            subprocess.run(["uglifyjs", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def simple_minify(js_code):
        return jsmin(js_code)

    @classmethod
    def obfuscate(cls, js_code):
        if cls.is_uglifyjs_available():
            with tempfile.NamedTemporaryFile(
                mode="w+", suffix=".js", delete=True
            ) as temp_file:
                temp_file.write(js_code)
                temp_file.flush()

                for attempt in range(cls.MAX_RETRIES):
                    try:
                        random_options = cls.get_random_uglify_options()
                        result = subprocess.run(
                            cls.UGLIFYJS_COMMAND + random_options + [temp_file.name],
                            capture_output=True,
                            text=True,
                            check=True,
                            timeout=3,
                        )
                        return result.stdout
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}")
                        logger.warning(f"UglifyJS stderr: {e.stderr}")
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            f"Attempt {attempt + 1} timed out after 5 seconds"
                        )

                    if attempt < cls.MAX_RETRIES - 1:
                        time.sleep(cls.RETRY_DELAY)
                    else:
                        logger.error(
                            f"All {cls.MAX_RETRIES} attempts to obfuscate with UglifyJS failed. Falling back to simple minification."
                        )
                        return cls.simple_minify(js_code)
        else:
            logger.warning("UglifyJS not found. Falling back to simple minification.")
            return cls.simple_minify(js_code)

    @staticmethod
    def get_random_uglify_options():
        options = []
        if random.choice([True, False]):
            options.append("--compress")
        if random.choice([True, False]):
            options.append("--mangle")
        if random.choice([True, False]):
            options.append("--mangle-props")
        if random.choice([True, False]):
            options.append("--toplevel")
        return options


def obfuscate_html_and_js(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Obfuscate JavaScript content
    for script in soup.find_all("script"):
        if script.string:
            obfuscated_js = JSObfuscator.obfuscate(script.string)
            script.string = obfuscated_js

    obfuscated_html = str(soup)
    return HTMLObfuscator.obfuscate(obfuscated_html)


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
    args = parser.parse_args()

    # Generate default output filename based on input filename
    input_filename = os.path.basename(args.input_file)
    input_name, input_ext = os.path.splitext(input_filename)
    output_file = args.output or f"{input_name}_obfuscated{input_ext}"

    process_file(args.input_file, output_file, obfuscate_html_and_js)


if __name__ == "__main__":
    main()
