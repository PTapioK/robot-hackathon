#!/usr/bin/env python3
"""
Auto-push script that commits and pushes changes to main every 5 minutes.
"""

import subprocess
import time
from datetime import datetime


def run_git_command(command):
    """Run a git command and return the output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        print(f"Error running command '{command}': {e}")
        return 1, "", str(e)


def has_changes():
    """Check if there are any changes to commit."""
    # Check for unstaged changes
    returncode, stdout, _ = run_git_command("git status --porcelain")
    return returncode == 0 and len(stdout) > 0


def commit_and_push():
    """Add, commit, and push changes if they exist."""
    if not has_changes():
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No changes to commit")
        return False

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Changes detected, committing...")

    # Add all changes
    returncode, stdout, stderr = run_git_command("git add .")
    if returncode != 0:
        print(f"Error adding changes: {stderr}")
        return False

    # Commit with timestamp
    commit_message = f"Auto-commit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    returncode, stdout, stderr = run_git_command(f'git commit -m "{commit_message}"')
    if returncode != 0:
        print(f"Error committing: {stderr}")
        return False

    print(f"  Committed: {commit_message}")

    # Push to main
    returncode, stdout, stderr = run_git_command("git push origin master")
    if returncode != 0:
        print(f"Error pushing: {stderr}")
        return False

    print(f"  Pushed to master successfully")
    return True


def main():
    """Main loop that runs every 5 minutes."""
    print("Auto-push script started. Checking for changes every 5 minutes...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            commit_and_push()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waiting 5 minutes...\n")
            time.sleep(300)  # 5 minutes = 300 seconds
    except KeyboardInterrupt:
        print("\n\nAuto-push script stopped.")


if __name__ == "__main__":
    main()
