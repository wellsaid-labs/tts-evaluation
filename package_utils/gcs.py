import shutil
import subprocess

gsutil = shutil.which("gsutil")


def gcs_cp(src, dst) -> str:
    cmd = f"{gsutil} cp {src} {dst}"
    subprocess.run(cmd.split(), check=True)
    return f"Successfully copied {src} to {dst}"
