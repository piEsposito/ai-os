#!/usr/bin/env python3
# This is https://github.com/VictorTaelin/AI-scripts/blob/main/chatsh.mjs but in Python.

import argparse
import os
import re
import subprocess
import sys

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
"""

DEFAULT_CONFIG_FILE = "pi_ai_os.yaml"
ALLOWED_EXTENSIONS = [".txt", ".md", ".py", ".sh", ".json", ".go", ".rs", ".js"]
IGNORED_DIRS = [
    ".git",
    ".idea",
    "__pycache__",
    "__pycache__",
    "node_modules",
    "venv",
    "assets",
]
ASSISTANT_MODEL_NAME = (
    "claude-3-haiku-20240307"  # hardcoded because it is fast and cheap af
)


ASSISTANT_MODEL_SYSTEM_PROMPT = """
# INPUT
You will be given a user prompt and a list of possibly relevant files on the current directory.

# OUTPUT
You must answer with a list of files that are, or that you predict WILL BE
used, directly or not, to complete the user prompt. Answer in a <FILES/> tag.
Optionally, you can also include a SHORT, 1-paragraph <JUSTIFICATION/>.

# EXAMPLE INPUT
<USER_PROMPT>
Create a script that receives two user inputs and sums them.
</USER_PROMPT>
<CONTEXT>
File: src/sum.py
Content:
```
def sum(a: int, b: int) -> int:
    return a + b
```
File: src/divide.py
Content:
```
def divide(a: int, b: int) -> int:
    return a / b
```
</CONTEXT>

</EXAMPLE_INPUT>
# EXAMPLE OUTPUT
<JUSTIFICATION>
The user will use the sum function to process the user inputs as per requested on the prompt.
</JUSTIFICATION>
<FILES>
src/sum.py
</FILES>

the FILES tag must contain the name of the relevant files, one per line, without anything else.
"""


def create_context_from_chdir():
    context = ""
    base_dir = "."
    for root, _, files in os.walk(base_dir):
        if any(root.endswith(dir) for dir in IGNORED_DIRS):
            continue
        for file in files:
            if not any(file.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue
            full_fname = os.path.join(root, file)
            with open(full_fname, "r") as f:
                file_content = f.read()
            context += f"File: {full_fname}\nContent:\n```{file_content}```\n\n"
    return context


def extract_dependencies(raw_text):
    pattern = r"<FILES>(.*?)</FILES>"
    match = re.search(pattern, raw_text, re.DOTALL)

    if match:
        content = match.group(1)
        return [line.strip() for line in content.split("\n") if line.strip()]
    else:
        return []


def select_relevant_files_on_chdir(user_prompt: str):
    assistant = AI(
        ASSISTANT_MODEL_NAME,
        system=ASSISTANT_MODEL_SYSTEM_PROMPT,
        max_new_tokens=1024,
        temperature=0,
    )
    context = create_context_from_chdir()
    prompt = f"<USER_PROMPT>{user_prompt}</USER_PROMPT>\n<CONTEXT>{context}</CONTEXT>"
    if os.environ.get("VERBOSE"):
        print(prompt)
    response = assistant(prompt)
    if os.environ.get("VERBOSE"):
        print(response)
    dependencies = extract_dependencies(response)
    if os.environ.get("VERBOSE"):
        print(f"Dependencies: {dependencies}")
    dependencies_content = ""
    for dependency in dependencies:
        with open(dependency, "r") as f:
            dependencies_content += (
                f"File: {dependency}\nContent:\n```{f.read()}```\n\n"
            )
    dependencies_context = f"<DEPENDENCIES>{dependencies_content}</DEPENDENCIES>"
    return dependencies_context


def extract_code(text: str) -> str:
    match = re.search(r"```sh([\s\S]*?)```", text)
    return match.group(1).strip() if match else None


def pi_ai_os(model: str, initial_message: str, config_file: str, assistant: bool):
    print(f"Welcome to AIOS. Model: {model}\n")

    config_file = config_file or DEFAULT_CONFIG_FILE
    if not os.path.exists(config_file):
        config_file = None

    system_prompt = SYSTEM_PROMPT.format(shell_info=get_shell_info())
    if os.environ.get("VERBOSE"):
        print(f"System prompt: {system_prompt}")

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
            if user_message.strip() == "/clear" or user_message.strip() == "/c":
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

        if assistant:
            dependencies_context = select_relevant_files_on_chdir(user_message)
            user_message = f"{user_message}\n\nYou are also provided following context: {dependencies_context}"

        if os.environ.get("VERBOSE"):
            print(f"User message: {user_message}")

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
    parser.add_argument(
        "-a",
        "--assistant",
        help="Triggers using an assistant model for context gathering",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase output verbosity",
        action="store_true",
    )
    parser.add_argument("message", nargs="*", help="Initial message to send to the AI")

    args = parser.parse_args()
    if args.verbose:
        os.environ["VERBOSE"] = "1"

    initial_message = " ".join(args.message) if args.message else None
    try:
        pi_ai_os(args.model, initial_message, args.config, args.assistant)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    main()
