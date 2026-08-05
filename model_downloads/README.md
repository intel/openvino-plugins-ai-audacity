# Model Download Manifests

This directory is the scaffold for generated `mod-openvino` download metadata.

Goals:

- Keep the human-maintained source of truth in concise JSON manifests.
- Avoid hand-maintaining SHA-256 values in C++.
- Generate a C++ header that `OVModelManagerPopulation.cpp` can consume later.
- Keep network access out of normal CMake builds.

Layout:

- `manifests/`: human-maintained model download metadata.
- `model_download_lock.json`: machine-maintained SHA-256 and size data.
- `gen_model_download_manifest_header.py`: refreshes the lock file and generates the C++ header.

Recommended workflow:

1. Edit manifests in `manifests/`.
2. Refresh hashes explicitly:

   `python model_downloads/gen_model_download_manifest_header.py refresh model_downloads/manifests model_downloads/model_download_lock.json`

3. Regenerate the header (uses `model_download_lock.json` by default when present):

   `python model_downloads/gen_model_download_manifest_header.py generate model_downloads/manifests mod-openvino/model_download_manifest_info.h`

4. Optional: generate without checksum injection:

   `python model_downloads/gen_model_download_manifest_header.py generate model_downloads/manifests mod-openvino/model_download_manifest_info.h --no-lock`

Notes:

- File sets are reusable name templates only. Checksums live in the lock file per resolved model file, because identical file names can map to different content under different source subdirectories.
- The generated header is intended to be committed, similar to the existing model card header flow.