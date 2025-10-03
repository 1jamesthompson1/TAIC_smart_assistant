
# TAIC Smart Assistant

> An intelligent assistant and knowledge search tool for the New Zealand Transport Accident Investigation Commission

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Built with Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange)](https://gradio.app/)

---

## Preview

![TAIC Smart Assistant Demo](https://github.com/1jamesthompson1/TAIC_smart_assistant/releases/download/v0.3.0/demo.gif)

---

## Features

🤖 **AI-Powered Assistant**
- Interactive chat interface with conversational AI
- Context-aware responses using TAIC's knowledge base
- Conversation history and management

🔍 **Advanced Knowledge Search**
- Vector-based semantic search across TAIC, ATSB and TSB documents
- Filter by document type, agency, year, and transport mode
- Visual analytics and result export capabilities

---

## Project Status

This chat interface has proven to be significantly more user-friendly than other methods. This means that as many tools will be built behind the chat interface.

As of October 2025 this app is going into organisation wide deployment. There is no public access at the moment.

The app implementation is completely seperate from the code base, however a new deployment would still require some admin and setup to get running.

## Contributing

### Setup for Local Development

_Note that this has only ever been 'completed' on Linux machines. For ease of use I would recommend [WSL](https://learn.microsoft.com/en-us/windows/wsl/about) for windows users_

1. **Install dependencies**

```bash
# Install uv if you haven't already
curl -Ls https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

2. **Get the vector database**

Install `azcopy` by following: https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10?tabs=apt
Install `az cli` by following: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

```bash
# Download and setup the vector database
uv run python working_files/download_vector_db.py
```

3. **Environment configuration**

```bash
# Copy and configure your environment variables
cp .env.example .env
# Edit .env with your Azure credentials and API keys
```

4. **Run the app locally**

```bash
uv run uvicorn app:app --host localhost --port 7860 --reload
```

5. **Access the application**
- Open your browser to `http://localhost:7860`
- For the tools interface: `http://localhost:7860/tools`

### Development Workflow

1. Create a feature branch from `main`
2. Make your changes with appropriate commit messages
3. Create a pull request
4. When merging, use squash-and-merge with conventional commit keywords in the final commit message for automatic version bumping

#### Version Management

This project uses semantic versioning (MAJOR.MINOR.PATCH):

- **Patch** versions are auto-bumped on every push to main
- **Minor/Major** versions can be manually triggered using GitHub Actions
- All conversations and searches store the app version they were created with
- Version compatibility checking prevents loading incompatible data

##### Auto-versioning Rules

Commit messages can trigger different version bumps:

- `BREAKING:` or `breaking change:` → Major version bump
- `feat:` or `feature:` or `minor:` → Minor version bump  
- Everything else → Patch version bump

##### Manual Version Bumping

Use GitHub Actions workflow "Auto Version Bump" with manual trigger to specify version bump type.

**Note for Pull Requests**: When using squash-and-merge, ensure your final squashed commit message contains the appropriate keywords above, as the GitHub Action analyzes the squashed commit message for version bumping.
