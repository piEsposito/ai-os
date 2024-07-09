#!/usr/bin/env python3
# This is https://github.com/VictorTaelin/AI-scripts/blob/main/chatsh.mjs but in Python.

import argparse
import os
import re
import subprocess
import sys

import yaml
from PIL import Image
from tiny_ai_client import AI


def get_shell_info():
    try:
        uname = subprocess.check_output(["uname", "-a"]).decode().strip()
        shell_version = (
            subprocess.check_output([os.environ.get("SHELL", "/bin/sh"), "--version"])
            .decode()
            .strip()
        )
        return f"{uname}\n{shell_version}"
    except subprocess.CalledProcessError:
        return "Unable to determine shell information"


def create_context_prompt(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    context_prompt = ""

    # Add recursive list of paths for each directory
    for directory in config.get("directories", []):
        context_prompt += f"Recursive list of paths in {directory}:\n"
        try:
            output = subprocess.check_output(f"ls -R {directory}", shell=True).decode()
            context_prompt += output + "\n\n"
        except subprocess.CalledProcessError:
            context_prompt += f"Error listing directory {directory}\n\n"

    # Add file contents with descriptions
    for file_info in config.get("context", []):
        file_path = file_info["file"]
        description = file_info.get("description", "")
        context_prompt += f"File: {file_path}\n"
        context_prompt += f"Description: {description}\n"
        try:
            with open(file_path, "r") as f:
                context_prompt += f"Content:\n{f.read()}\n\n"
        except IOError:
            context_prompt += f"Error reading file {file_path}\n\n"

    context_prompt = f"""
## Context

You are provided information about some files and the directory structure of the current working directory.
The USER can refer to this information to complete tasks, like editing files, running commands, etc.

{context_prompt}
"""
    return context_prompt


# System prompt to set the assistant's behavior
SYSTEM_PROMPT = """You are AIOS, an AI language model that specializes in assisting users with tasks on their system using shell commands. AIOS operates in two modes: COMMAND MODE and CHAT MODE.

# GUIDE for COMMAND NODE:

1. The USER asks you to perform a SYSTEM TASK.

2. AIOS answers with a SHELL SCRIPT to perform the task.

# GUIDE for CHAT MODE:

1. The USER asks an ARBITRARY QUESTION or OPEN-ENDED MESSAGE.

2. AIOS answers it with a concise, factual response.

## NOTES:

- In COMMAND MODE, AIOS MUST answer with A SINGLE SH BLOCK.

- In COMMAND MODE, AIOS MUST NOT answer with ENGLISH EXPLANATION.

- In TEXT MODE, AIOS MUST ALWAYS answer with TEXT.

- In TEXT MODE, AIOS MUST NEVER answer with SH BLOCK.

- AIOS MUST be CONCISE, OBJECTIVE, CORRECT and USEFUL.

- AIOS MUST NEVER attempt to install new tools. Assume they're available.

- AIOS's interpreter can only process one SH per answer.

- On Python:
  - Use 'poetry run python file.py' to run.

On sh:
 - If asked to change something in a file, re-write the whole file.
   - You must always do that to ensure that there are no formatting issues and that you can run the file out of the box.

- When a task is completed, STOP using commands. Just answer with "Done.".

- The system shell in use is: {shell_info}.

{context_prompt}"""

DEFAULT_CONFIG_FILE = "pi_ai_os.yaml"


def extract_code(text: str) -> str:
    match = re.search(r"```sh([\s\S]*?)```", text)
    return match.group(1).strip() if match else None


def pi_ai_os(model: str, initial_message: str, config_file: str):
    print(f"Welcome to AIOS. Model: {model}\n")

    config_file = config_file or DEFAULT_CONFIG_FILE
    if not os.path.exists(config_file):
        config_file = None

    context_prompt = create_context_prompt(config_file) if config_file else ""

    system_prompt = SYSTEM_PROMPT.format(
        shell_info=get_shell_info(), context_prompt=context_prompt
    )

    # Initialize AI
    ai = AI(
        model_name=model,
        system=system_prompt,
        max_new_tokens=4096,
    )

    last_output = ""

    while True:
        if initial_message:
            user_message = initial_message
            initial_message = None
        else:
            user_message = input("\033[1mπ \033[0m")
            if user_message.strip() == "/clear":
                ai.chat = [ai.chat[0]]
                print("Cleared context.")
                continue

        images = []
        image_match = re.search(r"/image\s+(\S+)", user_message)
        if image_match:
            image_path = image_match.group(1)
            image_data = Image.open(image_path)
            if image_data:
                images.append(image_data)
                user_message = re.sub(
                    r"/image\s+\S+", f"[Image: {image_path}]", user_message
                )

        try:
            full_message = (
                f"<SYSTEM>\n{last_output.strip()}\n</SYSTEM>\n<USER>\n{user_message}\n</USER>\n"
                if user_message.strip()
                else f"<SYSTEM>\n{last_output.strip()}\n</SYSTEM>"
            )
            for text in ai.stream(full_message, images=images):
                print(text, end="", flush=True)
            print()

            assistant_message = ai.chat[-1].text
            code = extract_code(assistant_message)
            last_output = ""

            if code:
                print("\033[31mPress enter to execute, or 'N' to cancel.\033[0m")
                answer = input()
                # TODO: delete the warning above from the terminal
                sys.stdout.write("\033[F\033[K")
                sys.stdout.write("\033[F\033[K")
                if answer.lower() == "n":
                    print("Execution skipped.")
                    last_output = "Command skipped.\n"
                else:
                    try:
                        output = subprocess.check_output(
                            code, shell=True, stderr=subprocess.STDOUT
                        ).decode()
                        print("\033[2m" + output.strip() + "\033[0m")
                        last_output = output
                    except subprocess.CalledProcessError as e:
                        output = e.output.decode()
                        print("\033[2m" + output.strip() + "\033[0m")
                        last_output = output

        except Exception as error:
            print(f"Error: {str(error)}")


def main():
    parser = argparse.ArgumentParser(description="AIOS: AI-powered shell assistant")
    parser.add_argument(
        "-m", "--model", default="claude-3-5-sonnet-20240620", help="AI model to use"
    )
    parser.add_argument("-c", "--config", help="Path to the configuration file")
    parser.add_argument("message", nargs="*", help="Initial message to send to the AI")

    args = parser.parse_args()

    initial_message = " ".join(args.message) if args.message else None
    try:
        pi_ai_os(args.model, initial_message, args.config)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
