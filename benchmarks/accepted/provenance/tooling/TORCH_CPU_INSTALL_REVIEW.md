# Read-only CPU wheel review

PASS for availability: the official PyTorch CPU index lists `torch 2.14.1+cpu` wheels for CPython 3.12 on Windows amd64 and Linux x86_64. Both object HEAD requests returned HTTP 200 on 2026-10-08 01:46:59 UTC. No wheel bodies were read or downloaded. Complete URLs, published index hash fragments and response headers are recorded in `torch_cpu_wheel_HEAD_receipt.json`; `torch_cpu_install_review.json` binds the evidence to the inspected workflow, requirements and documentation hashes.

The proposed CI-only command, inserted after the setuptools step and before the requirements file, is:

```sh
python -m pip install "torch==2.14.1+cpu" --index-url https://download.pytorch.org/whl/cpu
```

The subsequent existing `torch==2.14.1` requirement accepts the installed `2.14.1+cpu` version. This follows the [official packaging version-specifier rules](https://packaging.python.org/en/latest/specifications/version-specifiers/) and was checked with the installed `packaging` library. The current requirements step uses no upgrade flag.

[PyTorch's official previous-version installation examples](https://pytorch.org/get-started/previous-versions/) use the CPU index for Linux and Windows. The [official local-install page](https://pytorch.org/get-started/locally/) provides CPU choices for both operating systems; exact 2.14.1 availability here comes from the [official CPU wheel index](https://download.pytorch.org/whl/cpu/torch/) and object HEAD responses.

| Platform | Wheel tag | HEAD content length |
|---|---|---:|
| Linux x86_64 | cp312-cp312-manylinux_2_28_x86_64 | 196,253,677 bytes |
| Windows amd64 | cp312-cp312-win_amd64 | 123,996,008 bytes |

The Linux wheel requires glibc 2.28 or later. HEAD establishes availability at the recorded time, not payload integrity: published index SHA-256 fragments were not verified against wheel bytes. Dependency resolution, installation, hosted tests and default-PyPI wheel dependency metadata were not executed or inspected. The proposed explicit CPU variant avoids relying on the default index to choose the intended build, while the CPU payload remains substantial.

No workflow or package files were edited. This CI build selection does not establish parity with the frozen research runtime signature or authorize research-run resumption under a changed environment.
