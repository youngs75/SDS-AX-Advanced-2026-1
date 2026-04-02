"""
Custom Sandbox Backend Creation

Demonstrates:
- SandboxBackendProtocol interface (execute, download_files, upload_files, id)
- BaseSandbox pattern: subclass only needs core methods, file ops auto-provided
- Custom sandbox example simulating command execution
- 3 provider comparison: Runloop (67L), Modal (99L), Daytona (110L)
- Error filtering: TTY noise removal (cannot set terminal process group)
- Base64 + heredoc pattern: bypass ARG_MAX shell limits
- Integration: create_deep_agent(backend=sandbox) → execute tool automatic
"""

import asyncio
import base64
from typing import TypedDict, Optional, List
from langchain_core.chat_models.fake import FakeListChatModel

# from langchain_deepagents import create_deep_agent
# from langchain_deepagents.sandbox import BaseSandbox


# ==== Response Types ====

class ExecuteResponse(TypedDict):
    """Response from sandbox command execution"""
    stdout: str
    stderr: str
    exit_code: int


class FileDownloadResponse(TypedDict):
    """Response from sandbox file download"""
    path: str
    content: bytes


class FileUploadResponse(TypedDict):
    """Response from sandbox file upload"""
    path: str
    success: bool


# ==== SandboxBackendProtocol Interface ====

# Protocol requirements:
# 1. async def execute(command: str, timeout: float = 30.0) -> ExecuteResponse
# 2. async def download_files(paths: List[str]) -> List[FileDownloadResponse]
# 3. async def upload_files(files: List[tuple[str, bytes]]) -> List[FileUploadResponse]
# 4. @property id -> str: unique sandbox identifier


# ==== Custom Sandbox Implementation ====

class CustomSandbox:
    """
    Example custom sandbox backend.

    Demonstrates minimal interface needed to create a sandbox backend:
    - execute(): run commands and return output
    - download_files(): retrieve files from sandbox
    - upload_files(): send files to sandbox
    - id property: unique identifier
    """

    def __init__(self, sandbox_id: str = "custom-sandbox-001"):
        self._id = sandbox_id
        self._command_history: List[str] = []

    @property
    def id(self) -> str:
        """Unique sandbox identifier"""
        return self._id

    async def execute(self, command: str, timeout: float = 30.0) -> ExecuteResponse:
        """
        Execute command in sandbox.

        In production:
        - Send command to remote sandbox via API
        - Handle timeout enforcement
        - Filter TTY noise from stderr
        - Return structured response
        """
        self._command_history.append(command)

        # Simulate execution
        if "echo" in command:
            return ExecuteResponse(
                stdout=f"Executed: {command}\n",
                stderr="",
                exit_code=0
            )
        elif "error" in command:
            return ExecuteResponse(
                stdout="",
                stderr="Command failed: error keyword detected\n",
                exit_code=1
            )
        else:
            return ExecuteResponse(
                stdout=f"Command executed: {command}\n",
                stderr="",
                exit_code=0
            )

    async def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        """
        Download files from sandbox.

        In production:
        - Fetch files via API/SFTP
        - Handle large file streaming
        - Return binary content
        """
        results = []
        for path in paths:
            # Simulate file download
            content = f"Content of {path}".encode('utf-8')
            results.append(FileDownloadResponse(
                path=path,
                content=content
            ))
        return results

    async def upload_files(self, files: List[tuple[str, bytes]]) -> List[FileUploadResponse]:
        """
        Upload files to sandbox.

        In production:
        - Use base64 + heredoc to bypass ARG_MAX limits
        - Send via API/SFTP
        - Verify upload success
        """
        results = []
        for path, content in files:
            # Simulate file upload
            results.append(FileUploadResponse(
                path=path,
                success=True
            ))
        return results


# ==== BaseSandbox Pattern ====

# BaseSandbox provides:
# - File operations (read_file, write_file, list_directory)
# - Built on top of execute/download/upload primitives
# - Standardized error handling
#
# Subclass only needs to implement:
# - execute() - command execution
# - download_files() - file retrieval
# - upload_files() - file sending
# - id property - unique identifier
#
# All file operations automatically provided!


# ==== Provider Comparison ====

PROVIDER_COMPARISON = """
Provider Implementations (all inherit BaseSandbox):

1. Runloop (67 lines)
   - Smallest implementation
   - 30 minute default timeout
   - Direct API integration

2. Modal (99 lines)
   - Mid-size implementation
   - 30 minute default timeout
   - Spawns remote containers

3. Daytona (110 lines)
   - Largest implementation
   - 30 minute default timeout
   - Dev environment focus

All follow same pattern:
- Extend BaseSandbox
- Implement execute() + download/upload + id
- Get all file ops free
"""


# ==== Error Filtering ====

def filter_tty_noise(stderr: str) -> str:
    """
    Remove common TTY-related noise from stderr.

    Common noise patterns:
    - "cannot set terminal process group"
    - "no job control in this shell"
    - "[some pid] tcsetpgrp failed: Not a tty"

    These are harmless warnings when running in non-interactive environments.
    """
    noise_patterns = [
        "cannot set terminal process group",
        "no job control in this shell",
        "tcsetpgrp failed: Not a tty",
    ]

    lines = stderr.split('\n')
    filtered = [
        line for line in lines
        if not any(pattern in line for pattern in noise_patterns)
    ]

    return '\n'.join(filtered)


# ==== Base64 + Heredoc Pattern ====

def create_upload_command(path: str, content: bytes) -> str:
    """
    Create shell command to upload file using base64 + heredoc.

    Bypasses ARG_MAX limits (typically 128KB) by using heredoc instead of
    passing content as command argument.

    Pattern:
    base64 -d <<'EOF' > /path/to/file
    <base64_content>
    EOF
    """
    b64_content = base64.b64encode(content).decode('utf-8')

    return f"""base64 -d <<'EOF' > {path}
{b64_content}
EOF"""


# ==== Model Configuration ====

model = FakeListChatModel(responses=[
    "I'll execute the command",
    "Command executed successfully"
])


# ==== Agent Setup (Commented) ====

# Create custom sandbox backend
# sandbox = CustomSandbox(sandbox_id="demo-sandbox")
#
# Create agent with custom sandbox
# agent = create_deep_agent(
#     model=model,
#     backend=sandbox,  # Custom sandbox backend
#     system_prompt="You are a coding assistant with sandbox access"
# )
#
# Agent automatically gets execute tool that uses sandbox.execute()


# ==== Main Demo ====

async def main():
    """Demonstrate custom sandbox patterns"""

    print("=" * 60)
    print("Custom Sandbox Backend Demo")
    print("=" * 60)

    # 1. Create custom sandbox
    print("\n1. Creating custom sandbox backend...")
    sandbox = CustomSandbox(sandbox_id="demo-001")
    print(f"   Sandbox ID: {sandbox.id}")

    # 2. Execute commands
    print("\n2. Executing commands...")
    response = await sandbox.execute("echo 'Hello from sandbox'")
    print(f"   stdout: {response['stdout'].strip()}")
    print(f"   exit_code: {response['exit_code']}")

    # 3. Test error case
    print("\n3. Testing error handling...")
    error_response = await sandbox.execute("error command")
    print(f"   stderr: {error_response['stderr'].strip()}")
    print(f"   exit_code: {error_response['exit_code']}")

    # 4. Upload file
    print("\n4. Uploading file...")
    file_content = b"Hello, world!"
    upload_results = await sandbox.upload_files([
        ("/tmp/test.txt", file_content)
    ])
    print(f"   Uploaded: {upload_results[0]['path']}")
    print(f"   Success: {upload_results[0]['success']}")

    # 5. Download file
    print("\n5. Downloading file...")
    download_results = await sandbox.download_files(["/tmp/test.txt"])
    print(f"   Downloaded: {download_results[0]['path']}")
    print(f"   Content: {download_results[0]['content'].decode('utf-8')}")

    # 6. Base64 + heredoc pattern
    print("\n6. Base64 + heredoc upload pattern...")
    content = b"Large content that bypasses ARG_MAX"
    command = create_upload_command("/tmp/large.txt", content)
    print(f"   Command preview: {command[:80]}...")

    # 7. TTY noise filtering
    print("\n7. TTY noise filtering...")
    noisy_stderr = "cannot set terminal process group\nReal error here\nno job control"
    filtered = filter_tty_noise(noisy_stderr)
    print(f"   Original: {noisy_stderr}")
    print(f"   Filtered: {filtered}")

    # 8. Provider comparison
    print("\n8. Provider Comparison:")
    print(PROVIDER_COMPARISON)

    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("- Custom sandbox = 4 methods (execute, download, upload, id)")
    print("- BaseSandbox provides all file ops automatically")
    print("- All providers use 30min timeout by default")
    print("- Base64 + heredoc bypasses ARG_MAX shell limits")
    print("- TTY noise filtering for cleaner error messages")
    print("- create_deep_agent(backend=sandbox) → execute tool automatic")


if __name__ == "__main__":
    asyncio.run(main())
