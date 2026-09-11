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

Command reference:

- `refresh`: download remote file metadata, compute SHA-256 and size, and update the lock file.
- `cleanup`: remove stale lock entries that no longer match the current manifest file names, without contacting the network.
- `list-model-ids`: print the current model ids directly from the manifests.
- `generate`: emit the C++ header from manifests and the current lock file.

Recommended workflow:

1. Edit manifests in `manifests/`.
2. Refresh hashes explicitly when model content may have changed:

   `python model_downloads/gen_model_download_manifest_header.py refresh model_downloads/manifests model_downloads/model_download_lock.json`

3. Optional: prune stale lock entries after manifest shape changes without recomputing hashes or downloading anything:

   `python model_downloads/gen_model_download_manifest_header.py cleanup model_downloads/manifests model_downloads/model_download_lock.json`

   To clean only specific models, repeat `--model-id`:

   `python model_downloads/gen_model_download_manifest_header.py cleanup model_downloads/manifests model_downloads/model_download_lock.json --model-id noise_suppression_deepfilternet2 --model-id noise_suppression_deepfilternet3`

4. Regenerate the header (uses `model_download_lock.json` by default when present):

   `python model_downloads/gen_model_download_manifest_header.py generate model_downloads/manifests mod-openvino/model_download_manifest_info.h`

5. Optional: list the current model ids:

   `python model_downloads/gen_model_download_manifest_header.py list-model-ids model_downloads/manifests`

   Add `--effect` to narrow the output to one effect, such as `Noise Suppression`.

6. Optional: generate without checksum injection:

   `python model_downloads/gen_model_download_manifest_header.py generate model_downloads/manifests mod-openvino/model_download_manifest_info.h --no-lock`

Notes:

- File sets are reusable name templates only. Checksums live in the lock file per resolved model file, because identical file names can map to different content under different URL subdirectories.
- `refresh` is for data changes. `cleanup` is for manifest-only shape changes such as renaming or removing file names.
- The generated header is intended to be committed, similar to the existing model card header flow.
