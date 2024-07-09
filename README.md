# AI-OS: Plug an LLM into your OS

This is basically https://github.com/VictorTaelin/AI-scripts/blob/main/chatsh.mjs but in Python, with some tweaks and new features (like images and configs)

Example usage:

- [Logs os me adding streaming to Gemini on tiny-ai-client](assets/add-streaming-to-gemini.md)

https://github.com/piEsposito/ai-os/assets/47679710/0c8bf030-963b-4a46-aae5-0b3925ed7c36

## Installation

1. Clone this repository.
2. Run `poetry install` in the project directory.
3. Alternatively `pip install pi-ai-os`

## Usage

```bash
aios [options] [initial message]
```

Options:

- `-m`, `--model`: Specify the AI model to use (default: claude-3-5-sonnet-20240620)
- `-c`, `--config`: Path to the configuration file

Example:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
aios -m claude-3-5-sonnet-20240620 "List all files in the current directory"
```

## Features

### Image Analysis

To include an image in your query, use the `/image` command followed by the path to the image:

```
/image path/to/your/image.jpg Describe this image
```

### Configuration File

AIOS can use a configuration file to provide context about your project. By default, it looks for a file named `.aios.yaml` in the current directory. You can specify a different config file using the `-c` option.

Example configuration file:

```yaml
directories:
  - /path/to/project

context:
  - file: pi_ai_os.py
    description: Main application file
  - file: README.md
    description: Configuration settings
```

### Clearing Context

Type `/clear` during your interaction with AIOS.
