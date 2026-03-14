import os
import shutil
import logging

logger = logging.getLogger(__name__)

# File extensions that indicate code/project files (not session artifacts)
_CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.m', '.sh', '.bash', '.yaml', '.yml', '.toml', '.json', '.xml',
    '.html', '.css', '.scss', '.sql', '.ipynb', '.md', '.txt', '.csv',
    '.cfg', '.ini', '.env', '.dockerfile', '.makefile',
}

_SKIP_DIRS = {'tmp', 'media', '__pycache__', '.git', 'node_modules'}


def has_code_files(data_path: str) -> bool:
    """Check if data_path contains code/project files (not just session artifacts).

    Walks the directory looking for files with code-related extensions,
    skipping hidden dirs and common non-code artifacts.
    """
    if not data_path or not os.path.isdir(data_path):
        return False
    for root, dirs, files in os.walk(data_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in _CODE_EXTENSIONS:
                return True
    return False


def zip_code_files(data_path: str) -> str | None:
    """Create a zip of data_path if it contains code files.

    Returns the zip path, or None if no code files are found or
    data_path is empty/missing.
    """
    if not has_code_files(data_path):
        return None
    zip_base = data_path.rstrip(os.sep)
    try:
        return shutil.make_archive(zip_base, 'zip', root_dir=data_path)
    except Exception as e:
        logger.error("Failed to create zip archive of %s: %s", data_path, e)
        return None
