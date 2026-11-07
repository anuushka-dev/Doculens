# DocuLens: Complete Development Environment Setup

**Platform:** Windows 11 + WSL2 + Ubuntu 22.04 LTS  
**Editor:** Visual Studio Code  
**Container runtime:** Docker Desktop with WSL2 integration  
**Python environment:** Python 3.12 managed by uv  
**Repository:** `https://github.com/anuushka-dev/Doculens`

This guide takes a fresh Windows machine from software installation to a working DocuLens development environment. It includes verification commands, expected results, application startup, and Docker checks.

---

## 1. Required software and version baseline

| Component | Version / requirement |
|---|---|
| Operating system | Windows 11, 64-bit |
| Linux environment | Ubuntu 22.04 LTS |
| Linux subsystem | WSL2, preferably latest |
| Code editor | Visual Studio Code Stable, Windows x64 |
| Docker Desktop | 4.98.0, latest release as of 7 November 2026 |
| Docker Engine | Previously verified: 29.8.2 |
| Docker Compose | Previously verified: v5.6.0 |
| Git | Installed inside Ubuntu |
| Python | 3.12 |
| uv | Latest compatible standalone release; Docker builder uses 0.13.1 |
| PyTorch | 2.14.1+cpu from the project lock file |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM runtime | Ollama for local Q&A and structured extraction |
| LLM model | `qwen3.5:2b` |

**Important version rules**

- The repository's `.python-version` specifies Python `3.12`.
- `pyproject.toml` declares the project dependencies.
- `uv.lock` records the exact resolved dependency versions. Use `uv sync --locked` rather than manually installing packages.
- PyTorch is pinned to its CPU build. This prevents unnecessary CUDA and NVIDIA libraries from being installed into the standard Docker image.
- VS Code and Docker Desktop can receive updates independently. Record their installed versions rather than assuming every machine has the same release.

---

## 2. Install Windows Subsystem for Linux (WSL2)

### Step 2.1: Open PowerShell as Administrator

Right-click the Windows Start button and open **Terminal (Admin)** or **PowerShell (Admin)**.

Check Windows:

```powershell
winver
```

Install WSL with Ubuntu 22.04:

```powershell
wsl --install -d Ubuntu-22.04
```

Restart Windows if requested.

### Step 2.2: Update WSL

Open PowerShell:

```powershell
wsl --update
wsl --set-default-version 2
```

Check the installation:

```powershell
wsl --version
wsl --status
wsl --list --verbose
```

Expected result:

- Ubuntu-22.04 appears in the distribution list.
- Its VERSION column shows `2`.
- WSL reports a Linux kernel version.

Example:

```text
NAME            STATE           VERSION
Ubuntu-22.04    Running         2
```

### Step 2.3: Initialize Ubuntu

Open Ubuntu 22.04 from the Windows Start menu.

The first launch asks you to create:

- A Linux username.
- A Linux password.

This Linux account is separate from your Windows account.

Inside Ubuntu, verify:

```bash
lsb_release -a
uname -r
uname -m
```

Expected:

- Distribution: Ubuntu 22.04 LTS.
- Kernel: contains `microsoft-standard-WSL2`.
- Architecture: `x86_64` on an Intel/AMD x64 machine.

Update Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
```

Install essential development tools:

```bash
sudo apt install -y git curl ca-certificates build-essential
```

Verify:

```bash
git --version
curl --version
```

**Important:** Install project dependencies and work inside Ubuntu's Linux filesystem. Keep the repository under `~/Projects/Doculens`, not under `/mnt/c/Users/...`.

---

## 3. Install Visual Studio Code

Download the stable Windows version:

https://code.visualstudio.com/Download

Choose:

**Windows → User Installer → x64**

Install using the default options.

### Install these VS Code extensions

Open VS Code → Extensions (`Ctrl + Shift + X`).

Install:

| Extension | Purpose |
|---|---|
| WSL | Connect VS Code to Ubuntu |
| Python | Python support |
| Pylance | Python language intelligence |
| Docker | Container and image management |
| Ruff | Python linting and formatting |

Extension names and publishers:

- `ms-vscode-remote.remote-wsl`
- `ms-python.python`
- `ms-python.vscode-pylance`
- `ms-azuretools.vscode-docker`
- `charliermarsh.ruff`

### Open VS Code from Ubuntu

After installation, return to Ubuntu.

You will open the repository using:

```bash
code .
```

VS Code should open with a WSL indicator in its bottom-left corner.

The important distinction is that VS Code runs on Windows, while the terminal, Python environment, Git operations, and project files run inside Ubuntu.

---

## 4. Install Docker Desktop

Download Docker Desktop for Windows x86-64:

https://www.docker.com/products/docker-desktop/

Use the official Windows installation instructions:

https://docs.docker.com/desktop/setup/install/windows-install/

Install Docker Desktop.

When the installer asks which backend to use, select **WSL 2**.

### Step 4.1: Configure Docker Desktop

Open Docker Desktop.

Go to:

**Settings → General**

Enable:

```text
Use the WSL 2 based engine
```

Next:

**Settings → Resources → WSL Integration**

Enable:

```text
Ubuntu-22.04
```

Apply the changes and restart Docker Desktop if requested.

### Step 4.2: Verify Docker from Ubuntu

Make sure Docker Desktop is running.

Open Ubuntu and execute:

```bash
docker --version
docker compose version
docker version
```

Expected:

- Docker CLI reports a version.
- Docker Compose reports a version.
- `docker version` displays both Client and Server sections.

Check the engine:

```bash
docker info --format '{{.OSType}}'
```

Expected:

```text
linux
```

### Step 4.3: Run a test container

```bash
docker run --rm hello-world
```

Expected final message:

```text
Hello from Docker!
```

This confirms that the Docker CLI, Docker Engine, and WSL integration can communicate.

**Do not install a separate Docker Engine inside Ubuntu.** Docker Desktop already supplies the engine through WSL integration.

---

## 5. Install uv and Python 3.12

DocuLens uses `uv` for Python version management, virtual environments, dependency installation, and lock-file synchronization.

Official uv documentation:

https://docs.astral.sh/uv/getting-started/installation/

### Step 5.1: Install uv inside Ubuntu

Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Reload the shell environment:

```bash
source "$HOME/.local/bin/env"
```

Verify:

```bash
uv --version
```

### Step 5.2: Install Python 3.12

```bash
uv python install 3.12
```

Check available Python installations:

```bash
uv python list
```

Verify the selected interpreter:

```bash
uv run --python 3.12 python --version
```

Expected:

```text
Python 3.12.x
```

The exact patch release may change as newer Python 3.12 maintenance releases become available.

---

## 6. Clone DocuLens from GitHub

Make a project directory:

```bash
mkdir -p ~/Projects
cd ~/Projects
```

Clone the Phase 3 branch:

```bash
git clone --branch phase3 --single-branch https://github.com/anuushka-dev/Doculens.git
```

Enter the repository:

```bash
cd Doculens
```

Check the branch:

```bash
git branch --show-current
```

Expected:

```text
phase3
```

Check the latest commit:

```bash
git log -1 --oneline
```

Check the repository status:

```bash
git status --short --branch
```

A clean clone should show a clean working tree, with the current branch tracking its remote branch.

### Open the project in VS Code

From the repository root:

```bash
code .
```

You should now see the DocuLens project in VS Code, connected to Ubuntu through WSL.

---

## 7. Install DocuLens dependencies using uv

Ensure you are inside the repository:

```bash
cd ~/Projects/Doculens
```

Check the project files:

```bash
ls -la
```

Important files:

```text
pyproject.toml
uv.lock
.python-version
Dockerfile
compose.yaml
app/
scripts/
tests/
sample_docs/
```

Check the lock file:

```bash
uv lock --check
```

Synchronize the environment:

```bash
uv sync --locked
```

This command:

- Reads `pyproject.toml`.
- Uses the exact dependency versions in `uv.lock`.
- Installs the required development dependencies.
- Creates the project virtual environment in `.venv`.

You do not need to manually install FastAPI, FAISS, PyTorch, NumPy, or the other project dependencies.

### Verify Python and the virtual environment

```bash
uv run python --version
uv run python -c "import sys; print(sys.executable)"
```

The executable path should point inside the project's `.venv` directory.

### Verify PyTorch

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected:

```text
PyTorch: 2.14.1+cpu
CUDA available: False
```

This is intentional for the current CPU-only dependency configuration.

Check the dependency tree:

```bash
uv tree --package torch
```

The tree should not contain the large CUDA Toolkit and NVIDIA package chain.

### Verify the dependencies are installed

```bash
uv run python -c "import fastapi, fitz, faiss, torch, sentence_transformers; print('Core dependencies imported successfully')"
```

Expected:

```text
Core dependencies imported successfully
```

---

## 8. Verify the DocuLens codebase

Run the automated tests:

```bash
uv run pytest -q
```

Expected:

- Pytest discovers the project tests.
- Tests finish without failures.
- The final summary reports the number passed.

Run Ruff:

```bash
uv run ruff check .
```

Expected:

```text
All checks passed!
```

For a separate retrieval test:

```bash
uv run pytest tests/phase2 -q
```

For Phase 3 tests:

```bash
uv run pytest tests/phase3 -q
```

The test count can change as new tests are added. A successful exit code and zero failures are the important checks.

---

## 9. Configure Ollama on Windows

Ollama is required for the local Phase 3 Q&A and structured-extraction workflows.

It is separate from the sentence-transformer embedding model.

Download Ollama for Windows:

https://ollama.com/download/windows

Install it and open PowerShell.

Check:

```powershell
ollama --version
```

Pull the required model:

```powershell
ollama pull qwen3.5:2b
```

Verify the downloaded model:

```powershell
ollama list
```

Expected model entry:

```text
qwen3.5:2b
```

Check the local Ollama API:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

The response should contain the installed model.

### Allow Ubuntu WSL to access Ollama

First, open Ubuntu and find its default gateway:

```bash
ip route show default
```

Example:

```text
default via 172.21.192.1 dev eth0
```

In this example, the Windows-side gateway is `172.21.192.1`.

If the Ollama desktop process is already listening on the default port, quit it before manually restarting the server on the WSL interface.

Open Windows PowerShell:

```powershell
$env:OLLAMA_HOST = "172.21.192.1:11434"
ollama serve
```

Replace the address with the gateway shown by your own WSL installation.

Keep this PowerShell window open while using the WSL API.

### Verify the connection from Ubuntu

Inside Ubuntu:

```bash
HOST_IP=$(ip route show default | awk '{print $3; exit}')
curl -fsS "http://${HOST_IP}:11434/api/tags"
```

Expected:

- HTTP request succeeds.
- Ollama returns JSON.
- The `qwen3.5:2b` model appears in the response.

### Create the local environment file

From the DocuLens repository root:

```bash
cp -n .env.example .env
```

Set the Ollama URL using the current WSL gateway:

```bash
HOST_IP=$(ip route show default | awk '{print $3; exit}')
sed -i "s|^DOCULENS_OLLAMA_BASE_URL=.*|DOCULENS_OLLAMA_BASE_URL=http://${HOST_IP}:11434|" .env
```

Inspect the Ollama settings:

```bash
grep '^DOCULENS_OLLAMA' .env
```

The base URL should resemble:

```text
DOCULENS_OLLAMA_BASE_URL=http://172.21.192.1:11434
DOCULENS_OLLAMA_MODEL=qwen3.5:2b
```

The gateway address is an example. Use the address returned by your own WSL route.

`.env` contains local settings and must not be committed to Git. The repository's `.gitignore` excludes it.

Verify the settings:

```bash
uv run python -c "from app.core.config import get_settings; s=get_settings(); print('Environment:', s.environment); print('Ollama URL:', s.ollama_base_url); print('Ollama model:', s.ollama_model)"
```

---

## 10. Start the DocuLens API

From the Ubuntu terminal:

```bash
cd ~/Projects/Doculens
```

Start the FastAPI development server:

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Expected startup output includes:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Keep this terminal open.

Open a second Ubuntu terminal and test the health endpoint:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "ok",
  "service": "doculens-api"
}
```

Open interactive API documentation:

http://127.0.0.1:8000/docs

Current API routes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health check |
| POST | `/documents/extract` | PDF ingestion and text extraction |
| POST | `/documents/ask` | Grounded PDF Q&A |
| POST | `/documents/extract-structured` | Schema-based metadata extraction |

### Optional Q&A smoke test

With Ollama running and the sample PDF available:

```bash
curl -sS --max-time 240 \
  -X POST http://127.0.0.1:8000/documents/ask \
  -F "file=@sample_docs/nist_sp_800_63_4.pdf;type=application/pdf" \
  -F "question=Which three main areas do these guidelines cover?"
```

A valid JSON response may have either `answered` or `insufficient_evidence` status. An insufficient-evidence response is an intentional application outcome, not necessarily an API failure.

---

## 11. Build and run DocuLens using Docker

Make sure Docker Desktop is running.

Inside Ubuntu:

```bash
cd ~/Projects/Doculens
```

### Step 11.1: Verify the dependency lock

```bash
uv lock --check
uv tree --package torch
```

Confirm the CPU-only PyTorch build is selected.

### Step 11.2: Validate Compose configuration

```bash
docker compose config --quiet
```

No output and a successful exit code mean the Compose configuration is valid.

### Step 11.3: Build the image

```bash
docker compose build --progress=plain
```

The Dockerfile uses a multi-stage build:

1. A builder stage installs Python dependencies with uv.
2. A runtime stage receives the installed virtual environment and application files.
3. The application runs as a non-root user.

The first build may take several minutes because the dependency packages still need to be downloaded.

The CPU-only PyTorch lock avoids downloading the unnecessary CUDA and NVIDIA libraries.

Subsequent builds can reuse Docker's cached layers. Do not routinely use `--no-cache`.

### Step 11.4: Verify the container's PyTorch installation

```bash
docker compose run --rm --no-deps api python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected:

```text
PyTorch: 2.14.1+cpu
CUDA available: False
```

### Step 11.5: Start the container

```bash
docker compose up -d
```

Inspect the running service:

```bash
docker compose ps
```

The API container should eventually show:

```text
Up
```

and, when the health check succeeds:

```text
healthy
```

Check the API from Ubuntu:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "ok",
  "service": "doculens-api"
}
```

Inspect application logs:

```bash
docker compose logs --tail=100 api
```

Follow logs continuously:

```bash
docker compose logs -f api
```

Press `Ctrl + C` to stop following the log output. This does not shut down the container.

### Step 11.6: Stop Docker Compose

```bash
docker compose down
```

This stops and removes the Compose containers and network.

Do not add `-v` unless you intentionally want to remove named volumes.

**Docker and Ollama are separate services.** The current Compose configuration verifies the API container and its health endpoint, but does not pass the WSL Ollama URL into the container. Docker-based Q&A and structured extraction require additional host-network configuration before they can reach the Windows Ollama service. A healthy Docker container alone does not prove model connectivity.

---

## 12. Everyday commands

### Git

```bash
git status
git branch --show-current
git branch -a
git log -5 --oneline
git pull --ff-only
```

Check remote configuration:

```bash
git remote -v
```

### uv and Python

```bash
uv --version
uv python list
uv run python --version
uv sync --locked
uv tree
```

### Tests and lint

```bash
uv run pytest -q
uv run ruff check .
```

### FastAPI

```bash
uv run uvicorn app.main:app --reload
```

### Docker

```bash
docker --version
docker version
docker compose version
docker ps
docker image ls
docker compose ps
docker compose logs --tail=100 api
docker compose build
docker compose up -d
docker compose down
```

### Ollama

Run in Windows PowerShell:

```powershell
ollama --version
ollama list
```

### WSL

Run in Windows PowerShell:

```powershell
wsl --status
wsl --version
wsl --list --verbose
```

---

## 13. Troubleshooting checklist

### `uv: command not found`

Reload the Ubuntu shell:

```bash
source "$HOME/.local/bin/env"
uv --version
```

### `python: command not found`

Check uv-managed Python:

```bash
uv python list
uv python install 3.12
uv run python --version
```

### `docker: command not found` inside Ubuntu

1. Start Docker Desktop.
2. Open Docker Desktop Settings.
3. Check that the WSL 2 engine is enabled.
4. Check that Ubuntu integration is enabled.
5. Restart the Ubuntu terminal.

### Docker reports that it cannot connect to the daemon

Check Docker Desktop first, then run:

```bash
docker version
docker info
```

If WSL itself is unresponsive, run this in Windows PowerShell:

```powershell
wsl --shutdown
```

Restart Docker Desktop and open Ubuntu again.

### `uv sync --locked` fails

Check that the lock file matches the project configuration:

```bash
uv lock --check
```

Do not randomly delete the lock file or manually replace package versions.

### Ollama connection fails

From Ubuntu:

```bash
HOST_IP=$(ip route show default | awk '{print $3; exit}')
curl -v "http://${HOST_IP}:11434/api/tags"
```

Check that:

- Ollama is running on Windows.
- The server is listening on the WSL-facing interface.
- The `.env` base URL uses the correct gateway.
- The model is present in `ollama list`.

### Docker build is taking a long time

Run:

```bash
docker compose build --progress=plain
```

Inspect which build layer is taking time.

Check the PyTorch dependency:

```bash
uv tree --package torch
```

It should show `2.14.1+cpu`, without the CUDA/NVIDIA dependency chain.

Avoid deleting Docker's entire cache as a first troubleshooting step.

---

## 14. Important recovery and data-safety rules

The following commands can remove development data. Do not run them casually:

```text
wsl --unregister Ubuntu-22.04
docker system prune --volumes
docker compose down -v
```

Before changing branches, resetting files, or reinstalling software, inspect:

```bash
git status --short --branch
```

Keep the repository inside Ubuntu:

```text
/home/<linux-user>/Projects/Doculens
```

Keep local environment settings in `.env`.

Keep `pyproject.toml` and `uv.lock` together in Git.

A fresh clone only contains commits that have been pushed to GitHub. Local uncommitted changes must be committed and pushed before they can be restored through a fresh clone.

---

## 15. Final environment verification

Run these commands from the DocuLens repository root.

```bash
echo "========== SYSTEM =========="
lsb_release -ds
uname -r

echo "========== GIT =========="
git --version
git branch --show-current
git status --short --branch

echo "========== UV / PYTHON =========="
uv --version
uv run python --version

echo "========== PYTORCH =========="
uv run python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo "========== DEPENDENCIES =========="
uv lock --check

echo "========== TESTS =========="
uv run pytest -q

echo "========== RUFF =========="
uv run ruff check .

echo "========== DOCKER =========="
docker --version
docker compose version
docker info --format '{{.OSType}}'

echo "========== DOCKER HEALTH CHECK =========="
docker compose config --quiet
```

The setup is ready for development when:

- Ubuntu is running under WSL2.
- VS Code opens the project in WSL mode.
- Docker Desktop is integrated with Ubuntu.
- Git can access the GitHub repository.
- `uv sync --locked` completes successfully.
- Python 3.12 is selected.
- PyTorch uses the CPU-only build.
- Tests pass.
- Ruff passes.
- Docker can build and run the API health check.
- Ollama is reachable from Ubuntu when local model inference is required.

**The most important reproducibility files are `pyproject.toml`, `uv.lock`, `.python-version`, `Dockerfile`, and `compose.yaml`. Preserve them in Git.**