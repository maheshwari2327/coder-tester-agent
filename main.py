"""CLI interface for Coder-Tester Agent."""

import sys
from agent import MAX_FIXES, build_graph


WARNING_BANNER = """
================================================================================
⚠️  STUDENT SAFETY WARNING:
Running automatically generated code is only safe for classroom demos.
This agent executes code locally on your machine using subprocess.
Do not run untrusted prompts or deploy this to an unsecured production environment.
================================================================================
"""


def run_task(graph, task: str):
    """Run a single coding task through the LangGraph agent and display progress."""
    initial_state = {
        "task": task,
        "code": "",
        "tests": "",
        "passed": False,
        "output": "",
        "fixes": 0
    }

    print(f"\n🚀 Starting task: {task}\n" + "-" * 60)

    final_state = dict(initial_state)

    try:
        for event in graph.stream(initial_state, stream_mode="updates"):
            for node_name, node_update in event.items():
                final_state.update(node_update)

                if node_name == "coder":
                    print("[coder] 📝 Python function generated.")
                elif node_name == "tester":
                    print("[tester] 🧪 4 assert statements generated.")
                elif node_name == "run_tests":
                    if node_update.get("passed"):
                        print("[run_tests] ✅ ALL TESTS PASSED!")
                    else:
                        print("[run_tests] ❌ Tests failed. Output snippet:")
                        output_snippet = node_update.get("output", "").strip()
                        if output_snippet:
                            for line in output_snippet.splitlines():
                                print(f"    │ {line}")
                        else:
                            print("    │ (no output captured)")
                elif node_name == "debugger":
                    fixes = node_update.get("fixes", final_state.get("fixes", 0))
                    print(f"[debugger] 🔧 Applied fix {fixes}/{MAX_FIXES}. Re-testing...")

    except KeyboardInterrupt:
        print("\n\n[aborted] Run interrupted by user.")
        return
    except Exception as e:
        print(f"\n[error] An error occurred during graph execution: {e}")
        return

    # Print final summary
    print("\n" + "=" * 60)
    if final_state.get("passed"):
        print("🎉 STATUS: PASSED")
    else:
        print("💥 STATUS: FAILED")
    print(f"Fix attempts: {final_state.get('fixes', 0)}/{MAX_FIXES}")
    print("=" * 60)

    print("\n--- FINAL CODE ---")
    print(final_state.get("code", "").strip() or "(no code generated)")

    print("\n--- FINAL TESTS ---")
    print(final_state.get("tests", "").strip() or "(no tests generated)")

    if not final_state.get("passed") and final_state.get("output"):
        print("\n--- LAST ERROR OUTPUT ---")
        print(final_state.get("output").strip())

    print("\n" + "-" * 60)


def main():
    """Main CLI interactive loop."""
    print(WARNING_BANNER)

    try:
        graph = build_graph()
    except Exception as e:
        print(f"Error initializing agent graph: {e}")
        sys.exit(1)

    print("Type your task below. Type 'exit' or 'quit' to end.\n")

    while True:
        try:
            task = input("Coding task: ").strip()
            if not task:
                continue
            if task.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            run_task(graph, task)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
