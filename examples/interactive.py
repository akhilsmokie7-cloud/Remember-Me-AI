import os
import sys
import time
import torch
import cmd
from remember_me.core.csnp import CSNPManager

# ANSI Colors for Matrix Aesthetic
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

class CognitiveShell(cmd.Cmd):
    intro = f"""
{GREEN}{BOLD}
╔══════════════════════════════════════════════════════════════╗
║  REMEMBER ME v1.0 - COHERENT STATE NETWORK PROTOCOL (CSNP)   ║
║  Identity: Bolt/Palette Fusion Interface                     ║
╚══════════════════════════════════════════════════════════════╝
{RESET}
Type 'help' or '?' to list commands.
Type 'quit' to exit.
Type any text to interact with the Memory Core.
    """
    prompt = f"{BLUE}USER > {RESET}"

    def __init__(self):
        super().__init__()
        print(f"{YELLOW}>>> INITIALIZING NEURAL KERNEL...{RESET}")
        # Initialize with a small limit to demonstrate compression easily
        self.memory = CSNPManager(context_limit=5)
        print(f"{GREEN}✓ KERNEL ACTIVE (Context Limit: 5){RESET}")

    def precmd(self, line):
        """Allow commands to start with / for chat-app familiarity."""
        if line.startswith('/'):
            return line[1:]
        return line

    def default(self, line):
        """Handle conversation input."""
        if not line.strip():
            return

        user_input = line

        # Simulate AI Response (Echo/Acknowledgement since no LLM is attached)
        ai_response = self._simulate_ai_response(user_input)

        print(f"{GREEN}AI   > {ai_response}{RESET}")

        # Update Memory State
        print(f"{YELLOW}[Thinking... updating Coherent State]{RESET}", end="\r")
        before_count = len(self.memory.text_buffer)

        self.memory.update_state(user_input, ai_response)

        after_count = len(self.memory.text_buffer)

        # Visual Feedback on Compression
        if after_count == self.memory.context_limit and before_count == self.memory.context_limit:
            print(f"{RED}[⚡ COMPRESSION TRIGGERED] Memory Full. Lowest mass vector evicted.{RESET}")
        else:
            print(f"{YELLOW}[✓ MEMORY CONSOLIDATED] Buffer: {after_count}/{self.memory.context_limit}{RESET}")

    def _simulate_ai_response(self, text):
        """Simple mock AI to keep the conversation flowing."""
        text = text.lower()
        if "hello" in text:
            return "Greetings. I am listening."
        elif "name" in text:
            return "I am the Coherent State Network."
        elif "save" in text:
            return "You can use the /save command to persist this state."
        else:
            return f"I have processed: '{text}'. This is now part of my identity."

    def do_status(self, arg):
        """Show the current internal state of the CSNP kernel."""
        print(f"\n{BOLD}--- SYSTEM STATUS ---{RESET}")
        print(f"Memories: {len(self.memory.text_buffer)} / {self.memory.context_limit}")
        print(f"Identity Vector Norm: {self.memory.identity_state.norm().item():.4f}")
        print(f"Merkle Root: {self.memory.chain.get_root_hash()}")
        print(f"{BOLD}--- ACTIVE CONTEXT ---{RESET}")
        print(self.memory.retrieve_context())
        print("----------------------\n")

    def do_save(self, arg):
        """Save the current memory state to a file. Usage: /save [filename]"""
        filename = arg if arg else "brain.pt"
        try:
            self.memory.save_state(filename)
            print(f"{GREEN}✓ State successfully saved to {filename}{RESET}")
        except Exception as e:
            print(f"{RED}Error saving state: {e}{RESET}")

    def do_load(self, arg):
        """Load a memory state from a file. Usage: /load [filename]"""
        filename = arg if arg else "brain.pt"
        try:
            self.memory.load_state(filename)
            print(f"{GREEN}✓ State successfully loaded from {filename}{RESET}")
        except Exception as e:
            print(f"{RED}Error loading state: {e}{RESET}")

    def do_reset(self, arg):
        """Wipe the memory core."""
        print(f"{RED}>>> WIPING MEMORY CORE...{RESET}")
        self.memory = CSNPManager(context_limit=5)
        print(f"{GREEN}✓ SYSTEM RESET{RESET}")

    def do_quit(self, arg):
        """Exit the interface."""
        print(f"{BLUE}Disconnecting...{RESET}")
        return True

if __name__ == "__main__":
    try:
        CognitiveShell().cmdloop()
    except KeyboardInterrupt:
        print(f"\n{BLUE}Disconnecting...{RESET}")
