import argparse
import hashlib
import json
import os
import sys
import urllib.request


DOWNLOAD_BUFFER_SIZE = 64 * 1024


def find_manifest_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in sorted(filenames):
            if filename.lower().endswith(".json"):
                yield os.path.join(dirpath, filename)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def join_url(base_url, suffix):
    if not suffix:
        return base_url
    if base_url.endswith("/"):
        base = base_url
    else:
        base = base_url + "/"
    return base + suffix.lstrip("/")


def ensure_trailing_slash(value):
    if value and not value.endswith("/"):
        return value + "/"
    return value


def sanitize_identifier(value):
    result = []
    for char in value:
        if char.isalnum():
            result.append(char)
        else:
            result.append("_")
    identifier = "".join(result)
    if identifier and identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier


def cpp_string_literal(value):
    escaped = value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")
    return f'"{escaped}"'


def load_manifests(manifest_dir):
    sources = {}
    file_sets = {}
    models = []

    for path in find_manifest_files(manifest_dir):
        data = load_json(path)
        local_sources = data.get("sources", {})
        local_file_sets = data.get("file_sets", {})
        local_models = data.get("models", [])

        for name, source in local_sources.items():
            if name in sources:
                raise ValueError(f"Duplicate source key '{name}' in {path}")
            sources[name] = source

        for name, file_set in local_file_sets.items():
            if name in file_sets:
                raise ValueError(f"Duplicate file_set key '{name}' in {path}")
            file_sets[name] = file_set

        for model in local_models:
            model_copy = dict(model)
            model_copy["__manifest_path"] = path
            models.append(model_copy)

    seen_model_ids = set()
    for model in models:
        model_id = model["id"]
        if model_id in seen_model_ids:
            raise ValueError(f"Duplicate model id '{model_id}'")
        seen_model_ids.add(model_id)

    return {
        "sources": sources,
        "file_sets": file_sets,
        "models": models,
    }


def normalize_file_entries(entries):
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"name": entry})
        elif isinstance(entry, dict):
            if "name" not in entry:
                raise ValueError("File entry objects must contain a 'name' field")
            normalized.append(dict(entry))
        else:
            raise ValueError("File entries must be either strings or objects")
    return normalized


def resolve_model_files(model, file_sets):
    if "file_set" in model and "files" in model:
        raise ValueError(f"Model '{model['id']}' cannot declare both 'file_set' and 'files'")

    if "file_set" in model:
        file_set_name = model["file_set"]
        if file_set_name not in file_sets:
            raise ValueError(f"Model '{model['id']}' references missing file_set '{file_set_name}'")
        entries = file_sets[file_set_name]
    else:
        entries = model.get("files", [])

    return normalize_file_entries(entries)


def resolve_models(manifest_data):
    resolved = []
    for model in manifest_data["models"]:
        if "source" in model and "base_url" in model:
            raise ValueError(f"Model '{model['id']}' cannot declare both 'source' and 'base_url'")

        if "source" in model:
            source_name = model["source"]
            if source_name not in manifest_data["sources"]:
                raise ValueError(f"Model '{model['id']}' references missing source '{source_name}'")
            base_url = manifest_data["sources"][source_name]["base_url"]
        else:
            base_url = model.get("base_url", "")

        subdir = model.get("subdir", "")
        if base_url and subdir:
            base_url = join_url(base_url, subdir)

        base_url = ensure_trailing_slash(base_url)

        resolved.append({
            "effect": model["effect"],
            "id": model["id"],
            "name": model["name"],
            "info_key": model["info_key"],
            "base_url": base_url,
            "post_url": model.get("post_url", "?download=true"),
            "relative_path": model["relative_path"],
            "dependencies": list(model.get("dependencies", [])),
            "files": resolve_model_files(model, manifest_data["file_sets"]),
        })

    return resolved


def load_lock_file(lock_file):
    if not lock_file or not os.path.exists(lock_file):
        return {"version": 1, "entries": {}}
    return load_json(lock_file)


def ensure_lock_entry(lock_data, model):
    entries = lock_data.setdefault("entries", {})
    entry = entries.setdefault(model["id"], {"base_url": model["base_url"], "post_url": model["post_url"], "files": {}})
    entry["base_url"] = model["base_url"]
    entry["post_url"] = model["post_url"]
    return entry


def compute_remote_file_hash(url, timeout):
    sha256 = hashlib.sha256()
    size = 0

    with urllib.request.urlopen(url, timeout=timeout) as response:
        while True:
            chunk = response.read(DOWNLOAD_BUFFER_SIZE)
            if not chunk:
                break
            sha256.update(chunk)
            size += len(chunk)

    return sha256.hexdigest(), size


def refresh_lock(manifest_dir, lock_file, timeout, force, selected_models):
    manifest_data = load_manifests(manifest_dir)
    models = resolve_models(manifest_data)
    lock_data = load_lock_file(lock_file)

    selected_model_ids = set(selected_models or [])

    for model in models:
        if selected_model_ids and model["id"] not in selected_model_ids:
            continue

        if not model["base_url"]:
            print(f"Skipping {model['id']}: no base_url configured", file=sys.stderr)
            continue

        lock_entry = ensure_lock_entry(lock_data, model)
        file_entries = lock_entry.setdefault("files", {})

        for file_info in model["files"]:
            file_name = file_info["name"]
            existing = file_entries.get(file_name, {})
            if existing.get("sha256") and not force:
                continue

            url = model["base_url"] + file_name + model["post_url"]
            print(f"Hashing {model['id']} -> {file_name}", file=sys.stderr)
            sha256, size = compute_remote_file_hash(url, timeout)
            file_entries[file_name] = {
                "sha256": sha256,
                "size": size,
                "url": url,
            }

    write_json(lock_file, lock_data)
    print(f"Lock file written to: {lock_file}")


def generate_header(manifest_dir, output_file, lock_file):
    manifest_data = load_manifests(manifest_dir)
    models = resolve_models(manifest_data)
    lock_data = load_lock_file(lock_file)
    lock_entries = lock_data.get("entries", {})

    lines = []
    lines.append("// Auto-generated header containing model download metadata")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cstddef>")
    lines.append("#include <cstdint>")
    lines.append("")
    lines.append("namespace model_download_manifest {")
    lines.append("")
    lines.append("struct ModelFileInfo")
    lines.append("{")
    lines.append("   const char* name;")
    lines.append("   const char* expected_sha256;")
    lines.append("   std::uint64_t expected_size;")
    lines.append("};")
    lines.append("")
    lines.append("struct ModelInfo")
    lines.append("{")
    lines.append("   const char* effect;")
    lines.append("   const char* model_id;")
    lines.append("   const char* model_name;")
    lines.append("   const char* info_key;")
    lines.append("   const char* base_url;")
    lines.append("   const char* post_url;")
    lines.append("   const char* relative_path;")
    lines.append("   const char* const* dependencies;")
    lines.append("   std::size_t dependency_count;")
    lines.append("   const ModelFileInfo* files;")
    lines.append("   std::size_t file_count;")
    lines.append("};")
    lines.append("")

    for model in models:
        model_identifier = sanitize_identifier(model["id"])
        lock_entry = lock_entries.get(model["id"], {})
        locked_files = lock_entry.get("files", {})

        lines.append(f"inline constexpr ModelFileInfo kFiles_{model_identifier}[] = {{")
        for file_info in model["files"]:
            file_name = file_info["name"]
            locked_file = locked_files.get(file_name, {})
            sha256 = locked_file.get("sha256", "")
            expected_size = locked_file.get("size", 0)
            if expected_size is None:
                expected_size = 0
            lines.append(f"   {{ {cpp_string_literal(file_name)}, {cpp_string_literal(sha256)}, {int(expected_size)}ULL }},")
        lines.append("};")
        lines.append("")

        dependencies = model["dependencies"]
        if dependencies:
            lines.append(f"inline constexpr const char* kDependencies_{model_identifier}[] = {{")
            for dependency in dependencies:
                lines.append(f"   {cpp_string_literal(dependency)},")
            lines.append("};")
        else:
            lines.append(f"inline constexpr const char* const* kDependencies_{model_identifier} = nullptr;")
        lines.append("")

    lines.append("inline constexpr ModelInfo kModels[] = {")
    for model in models:
        model_identifier = sanitize_identifier(model["id"])
        dependencies = model["dependencies"]
        dependency_ref = f"kDependencies_{model_identifier}"
        dependency_count = str(len(dependencies))
        lines.append("   {")
        lines.append(f"      {cpp_string_literal(model['effect'])},")
        lines.append(f"      {cpp_string_literal(model['id'])},")
        lines.append(f"      {cpp_string_literal(model['name'])},")
        lines.append(f"      {cpp_string_literal(model['info_key'])},")
        lines.append(f"      {cpp_string_literal(model['base_url'])},")
        lines.append(f"      {cpp_string_literal(model['post_url'])},")
        lines.append(f"      {cpp_string_literal(model['relative_path'])},")
        lines.append(f"      {dependency_ref},")
        lines.append(f"      {dependency_count},")
        lines.append(f"      kFiles_{model_identifier},")
        lines.append(f"      sizeof(kFiles_{model_identifier}) / sizeof(kFiles_{model_identifier}[0])")
        lines.append("   },")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr std::size_t kModelCount = sizeof(kModels) / sizeof(kModels[0]);")
    lines.append("")
    lines.append("} // namespace model_download_manifest")

    with open(output_file, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Header written to: {output_file} ({len(models)} models)")


def resolve_generate_lock_file(manifest_dir, explicit_lock_file, no_lock):
    if no_lock:
        return None

    if explicit_lock_file:
        return explicit_lock_file

    default_lock = os.path.normpath(os.path.join(os.path.abspath(manifest_dir), "..", "model_download_lock.json"))
    if os.path.exists(default_lock):
        print(f"Using default lock file: {default_lock}")
        return default_lock

    print("No lock file provided and no default model_download_lock.json found; generating without checksums.", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate model download metadata headers and refresh checksum lock data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate a C++ header from JSON manifests")
    generate_parser.add_argument("manifest_dir", help="Directory to search recursively for manifest JSON files")
    generate_parser.add_argument("output_file", help="Generated output header path")
    generate_parser.add_argument("--lock-file", default=None, help="Lock file containing resolved SHA-256 values")
    generate_parser.add_argument("--no-lock", action="store_true", help="Generate without lock-file checksums")

    refresh_parser = subparsers.add_parser("refresh", help="Download remote model files and refresh the checksum lock file")
    refresh_parser.add_argument("manifest_dir", help="Directory to search recursively for manifest JSON files")
    refresh_parser.add_argument("lock_file", help="Output lock file path")
    refresh_parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds")
    refresh_parser.add_argument("--force", action="store_true", help="Recompute hashes even if lock entries already exist")
    refresh_parser.add_argument("--model-id", action="append", default=[], help="Restrict refresh to specific model ids")

    args = parser.parse_args()

    if args.command == "generate":
        lock_file = resolve_generate_lock_file(args.manifest_dir, args.lock_file, args.no_lock)
        generate_header(args.manifest_dir, args.output_file, lock_file)
    elif args.command == "refresh":
        refresh_lock(args.manifest_dir, args.lock_file, args.timeout, args.force, args.model_id)


if __name__ == "__main__":
    main()