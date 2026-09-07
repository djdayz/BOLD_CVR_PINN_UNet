from __future__ import annotations


def identity_registration_warning() -> str:
    return (
        "Using identity registration fallback. This is suitable only for tests or already aligned "
        "synthetic data; real T1-to-BOLD registration should use FSL FLIRT or ANTs."
    )
