import sys
from edgent.loop import AgentLoop

def main():
    print("=" * 60)
    print("  Edgent - Autonomous AI Agent Harness")
    print("  Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        agent = AgentLoop()
    except Exception as e:
        print(f"[Initialization Error]: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("\nEdgent> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            response = agent.run(user_input)
            print(f"\n[Response]:\n{response}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting Edgent CLI.")
            break
        except Exception as e:
            print(f"\n[Runtime Error]: {e}")

if __name__ == "__main__":
    main()