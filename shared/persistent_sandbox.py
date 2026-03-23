"""Persistent Docker sandbox: one container per task, reused across all samples."""

import base64
import errno
import logging
import os
import tempfile
import uuid as uuid_mod
from pathlib import PurePosixPath
from typing import Literal, Union, overload

from typing_extensions import override

from inspect_ai.util._sandbox.environment import (
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from inspect_ai.util._sandbox.limits import (
    SandboxEnvironmentLimits,
    verify_read_file_size,
)
from inspect_ai.util._sandbox.registry import sandboxenv
from inspect_ai.util._subprocess import ExecResult, subprocess

logger = logging.getLogger(__name__)


def _short_uuid() -> str:
    return uuid_mod.uuid4().hex[:12]


@sandboxenv(name="persistent_docker")
class PersistentDockerSandbox(SandboxEnvironment):
    """A Docker sandbox that starts one container per task and reuses it for all samples."""

    # Class-level state: config -> container_name
    # Keyed by config (not task_name) because inspect_ai passes "startup"/"shutdown"
    # as task_name to task_init/task_cleanup, but the real task name to sample_init.
    _containers: dict[SandboxEnvironmentConfigType | None, str] = {}

    def __init__(self, container: str, working_dir: str) -> None:
        super().__init__()
        self._container = container
        self._working_dir = working_dir

    # ── Task lifecycle (once per task) ────────────────────────────────

    @override
    @classmethod
    async def task_init(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None
    ) -> None:
        image = config if isinstance(config, str) and config else "python:3.12-slim"
        container_name = f"inspect-persist-{_short_uuid()}"

        # Pull image
        logger.info("Pulling image %s", image)
        pull = await subprocess(["docker", "pull", image], timeout=300)
        if not pull.success:
            raise RuntimeError(f"Failed to pull image {image}: {pull.stderr}")

        # Start container
        logger.info("Starting persistent container %s", container_name)
        run = await subprocess(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--init",
                image,
                "sleep",
                "infinity",
            ],
            timeout=60,
        )
        if not run.success:
            raise RuntimeError(f"Failed to start container: {run.stderr}")

        cls._containers[config] = container_name

    @override
    @classmethod
    async def task_cleanup(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool
    ) -> None:
        container = cls._containers.pop(config, None)
        if cleanup and container:
            logger.info("Removing persistent container %s", container)
            await subprocess(["docker", "rm", "-f", container], timeout=30)

    # ── Sample lifecycle (per sample) ─────────────────────────────────

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        container = cls._containers[config]
        sample_dir = f"/tmp/sample_{_short_uuid()}"

        # Create per-sample working directory inside the container
        result = await subprocess(
            ["docker", "exec", container, "mkdir", "-p", sample_dir],
            timeout=10,
        )
        if not result.success:
            raise RuntimeError(
                f"Failed to create sample directory: {result.stderr}"
            )

        return {"default": PersistentDockerSandbox(container, sample_dir)}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        if interrupted:
            return
        env = next(iter(environments.values())).as_type(PersistentDockerSandbox)
        container = cls._containers.get(config)
        if container:
            await subprocess(
                ["docker", "exec", container, "rm", "-rf", env._working_dir],
                timeout=10,
            )

    # ── exec ──────────────────────────────────────────────────────────

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        args = ["docker", "exec"]

        # Working directory
        final_cwd = PurePosixPath(self._working_dir if cwd is None else cwd)
        if not final_cwd.is_absolute():
            final_cwd = PurePosixPath(self._working_dir) / final_cwd
        args.extend(["--workdir", str(final_cwd)])

        # User
        if user:
            args.extend(["--user", user])

        # Environment variables
        if env:
            for key, value in env.items():
                args.extend(["--env", f"{key}={value}"])

        # Interactive mode for stdin
        if input is not None:
            args.append("-i")

        args.append(self._container)
        args.extend(cmd)

        result = await subprocess(
            args=args,
            input=input,
            timeout=timeout,
            output_limit=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE,
            concurrency=concurrency,
        )

        if result.returncode == 126 and "permission denied" in result.stdout:
            raise PermissionError(f"Permission denied executing command: {result}")

        return result

    # ── write_file ────────────────────────────────────────────────────

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        file = self._resolve_path(file)

        # Ensure parent directory exists
        parent = PurePosixPath(file).parent.as_posix()
        if parent != ".":
            result = await self.exec(["mkdir", "-p", parent])
            if not result.success:
                raise RuntimeError(
                    f"Failed to create directory {parent}: {result.stderr}"
                )

        if isinstance(contents, str):
            result = await self.exec(
                ["sh", "-e", "-c", 'tee -- "$1" > /dev/null', "write", file],
                input=contents,
                timeout=180,
            )
        else:
            b64 = base64.b64encode(contents).decode("ascii")
            result = await self.exec(
                [
                    "sh",
                    "-e",
                    "-c",
                    'base64 -d | tee -- "$1" > /dev/null',
                    "write",
                    file,
                ],
                input=b64,
                timeout=180,
            )

        if result.returncode != 0:
            stderr = result.stderr.casefold()
            if "permission denied" in stderr:
                raise PermissionError(f"Permission denied writing {file}: {result.stderr}")
            elif "is a directory" in stderr or "cannot overwrite directory" in stderr:
                raise IsADirectoryError(f"{file} is a directory")
            else:
                raise RuntimeError(f"Failed to write file {file}: {result.stderr}")

    # ── read_file ─────────────────────────────────────────────────────

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> Union[str, bytes]:
        file = self._resolve_path(file)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dest = os.path.join(tmp, os.path.basename(file))

            result = await subprocess(
                ["docker", "cp", f"{self._container}:{file}", dest],
                timeout=120,
            )
            if not result.success:
                msg = result.stderr.lower()
                if "no such file" in msg or "could not find" in msg:
                    raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
                elif "permission denied" in msg:
                    raise PermissionError(errno.EACCES, "Permission denied", file)
                raise RuntimeError(f"Failed to read file {file}: {result.stderr}")

            verify_read_file_size(dest)

            if text:
                with open(dest, "r", newline="", encoding="utf-8") as f:
                    return f.read()
            else:
                with open(dest, "rb") as f:
                    return f.read()

    # ── helpers ───────────────────────────────────────────────────────

    def _resolve_path(self, file: str) -> str:
        path = PurePosixPath(file)
        if not path.is_absolute():
            path = PurePosixPath(self._working_dir) / path
        return path.as_posix()
