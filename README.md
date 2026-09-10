# LLM Lab

A hands-on learning project for understanding how Large Language Models (LLMs) work, from tokenization to building a Tiny LLM with PyTorch.

## Getting Started

### Stage 1 — Install Python

Download and install **Python 3.11.x or 3.12.x** for Windows:

https://www.python.org/downloads/windows/

During installation, enable:

```text
Add python.exe to PATH
```

Verify the installation:

```bash
python --version
python -m pip --version
```

### Stage 2 — Clone and Run the Project

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <PROJECT_DIRECTORY>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Run a lab script:

```bash
python <LAB_DIRECTORY>/<SCRIPT_NAME>.py
```

The `.venv` directory is local to your machine and should not be committed to Git.