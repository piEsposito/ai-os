# AIOS: give a LLM full access to your terminal for the sake of science.

This is basically https://github.com/VictorTaelin/AI-scripts/blob/main/chatsh.mjs implemented in Python, with additional features like image analysis, configuration files.

Example usage:

- [Logs of adding streaming to Gemini on tiny-ai-client](assets/add-streaming-to-gemini.md)

https://github.com/piEsposito/ai-os/assets/47679710/0c8bf030-963b-4a46-aae5-0b3925ed7c36

## Installation

1. Clone this repository.
2. Run `poetry install` in the project directory.
3. Alternatively, install via pip: `pip install pi-ai-os`

## Usage

```bash
aios [options] [initial message]
```

Options:

- `-m`, `--model`: Specify the AI model to use (default: claude-3-5-sonnet-20240620)
- `-c`, `--config`: Path to the configuration file
- `-a`, `--assistant`: Use an assistant model for context gathering, hardcoded to Claude Haiku for speed and cost
- `-v`, `--verbose`: Increase output verbosity

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

AIOS can use a configuration file to provide context about your project. By default, it looks for a file named `pi_ai_os.yaml` in the current directory. You can specify a different config file using the `-c` option.

Example configuration file:

```yaml
allowed_extensions:
  - .txt
  - .md
  - .py
ignored_dirs:
  - .git
  - node_modules
```

### Context-Aware Assistance

Use the `-a` or `--assistant` flag to enable context-aware assistance. This feature uses an assistant model to analyze the project structure and select relevant files for providing context to the main AI model.

### Clearing Context

Type `/clear` or `/c` during your interaction with AIOS to clear the conversation context.

### Verbose Mode

Use the `-v` or `--verbose` flag to increase output verbosity, which can be helpful for debugging or understanding the AI's decision-making process.
