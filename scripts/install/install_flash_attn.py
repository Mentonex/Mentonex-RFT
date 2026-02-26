"""This script is used to install flash-attn from a pre-built wheel hosted on an OSS bucket.
Useful for mainland China users who have difficulty installing flash-attn from PyPI due to network issues.
"""
import os
import platform
import subprocess
import sys
import tempfile

import torch
import typer

app = typer.Typer()
FLASH_VERSION = "2.8.1"


def check_flash_attn_installed():
    try:
        import flash_attn

        print(f"flash_attn version: {flash_attn.__version__}")
        return True
    except ImportError:
        return False


def install_flash_attn(uv: bool = False, keep_wheel: bool = False):
    # Get torch version
    TORCH_VERSION_RAW = torch.__version__
    torch_major, torch_minor = TORCH_VERSION_RAW.split(".")[:2]
    torch_version = f"{torch_major}.{torch_minor}"

    # Get python version
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    # Get platform name
    platform_name = platform.system().lower() + "_" + platform.machine()

    # Get cxx11_abi
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()

    # Is ROCM
    # torch.version.hip/cuda are runtime attributes not in type stubs
    IS_ROCM = hasattr(torch.version, "hip") and torch.version.hip is not None  # type: ignore[attr-defined]

    if IS_ROCM:
        print("We currently do not host ROCm wheels for flash-attn.")
        sys.exit(1)
    else:
        torch_cuda_version = torch.version.cuda  # type: ignore[attr-defined]
        cuda_major = torch_cuda_version.split(".")[0] if torch_cuda_version else None
        if cuda_major != "12":
            print("Only CUDA 12 wheels are hosted for flash-attn.")
            sys.exit(1)
        cuda_version = "12"
        wheel_filename = (
            f"flash_attn-{FLASH_VERSION}%2Bcu{cuda_version}torch{torch_version}"
            f"cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
        )
        local_filename = (
            f"flash_attn-{FLASH_VERSION}-{python_version}-{python_version}-{platform_name}.whl"
        )

    wheel_url = (
        "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com"
        f"/AgentScope/download/flash-attn/{FLASH_VERSION}/{wheel_filename}"
    )

    print(f"wheel_url: {wheel_url}")
    print(f"target_local_file: {local_filename}")

    def _install_helper(local_path: str):
        subprocess.run(["wget", wheel_url, "-O", local_path], check=True)
        install_cmd = (
            ["uv", "pip", "install", local_path]
            if uv
            else [sys.executable, "-m", "pip", "install", local_path]
        )
        subprocess.run(install_cmd, check=True)

    if keep_wheel:
        local_path = os.path.abspath(local_filename)
        _install_helper(local_path)
    else:
        with tempfile.TemporaryDirectory() as tempdir:
            local_path = os.path.join(tempdir, local_filename)
            _install_helper(local_path)

    # Try to import flash_attn
    if not check_flash_attn_installed():
        print("Failed to install flash_attn.")
        sys.exit(1)


@app.command()
def main(
    uv: bool = typer.Option(False, help="Use uv pip to install instead of pip"),
    keep_wheel: bool = typer.Option(
        False, help="Keep the downloaded wheel file in current directory"
    ),
):
    """Install flash-attn from a pre-built wheel."""
    if check_flash_attn_installed():
        print("flash_attn is already installed. Skipping installation.")
        return
    install_flash_attn(uv=uv, keep_wheel=keep_wheel)


if __name__ == "__main__":
    typer.run(main)
