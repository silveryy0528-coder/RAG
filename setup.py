from pathlib import Path

from setuptools import find_namespace_packages, setup

ROOT = Path(__file__).resolve().parent


def read_requirements(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


setup(
    # Project metadata
    name="rag",
    version="0.0.1",
    author="Yan Guo",
    description="A lightweight retrieval-augmented generation project built on FAISS and SentenceTransformers.",
    url="https://github.com/silveryy0528-coder/RAG",
    python_requires=">=3.10",

    # Which Python code gets packaged
    packages=find_namespace_packages(include=["rag", "rag.*", "scripts", "scripts.*"]),
    include_package_data=True,

    # Runtime and development dependencies
    install_requires=read_requirements(ROOT / "requirements.txt"),
    extras_require={
        "dev": [
            "pytest",
            "black",
        ],
    },

    # Create command-line scripts that can be run after installing the package
    entry_points={
        "console_scripts": [
            "rag-chat=scripts.rag_chat:main",
            "rag-build=scripts.build_index:main",
            "rag-evaluate=scripts.evaluate:main",
        ]
    },
)