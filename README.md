# Coder <-> Tester Multi-Agent System

An autonomous code generation, testing, and debugging multi-agent system implemented with **LangChain**, **LangGraph**, and **Ollama**.

Available as both a **Web UI (Streamlit)** and a **Terminal CLI**.

---

## Architecture & Flow

```
                      +-------------------+
                      |       START       |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |       coder       |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |      tester       |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                 +--->|     run_tests     |
                 |    +---------+---------+
                 |              |
                 |       (passed / fixes >= 3?)
                 |         /         \
                 |      No            Yes
                 |       v             v
          +------------+----+     +-------+
          |     debugger    |     |  END  |
          +-----------------+     +-------+
```

1. **Coder (`coder`)**: Writes a single Python function solving the requested task.
2. **Tester (`tester`)**: Writes 4 simple `assert` lines calling the function.
3. **Run Tests (`run_tests`)**: Subprocess execution environment running `code + tests + print('ALL TESTS PASSED')`.
4. **Debugger (`debugger`)**: Analyzes error output and repairs either the function or the tests when assertions fail. Loops up to `MAX_FIXES = 3`.

---

## Prerequisites

1. **Install Ollama** and pull the `llama3.2` model:
   ```bash
   ollama pull llama3.2
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🌐 How to Run in Website (Web UI)

Launch the interactive web application using Streamlit:

```bash
streamlit run app.py
```

Streamlit will start a local web server and automatically open your default browser at:
```
http://localhost:8501
```

### Web UI Features:
- **Live Agent Visualizer**: Displays each node's output (`coder`, `tester`, `run_tests`, `debugger`) in real-time as it executes.
- **Syntax Highlighting & Copy Buttons**: View formatted Python code and tests directly on the screen.
- **Simulation / Demo Mode Toggle**: Allows testing the complete multi-agent repair loop without needing an active Ollama instance.
- **Configurable Settings**: Change Ollama model and base URL directly from the sidebar.
- **Quick-Fill Example Tasks**: One-click preset prompts to test algorithmic problems.

---

## 💻 How to Run in Terminal CLI

If you prefer the command line:

```bash
python main.py
```
Or:
```bash
python agent.py
```

---

## 🧪 Running the Automated Tests

To run the unit test suite verifying tool execution, timeouts, and graph routing:

```bash
pytest
```
