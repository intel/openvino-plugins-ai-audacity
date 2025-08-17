import os
import argparse
import markdown

def sanitize_varname(path, root_dir):
    """
    Convert a relative file path to a safe C++ variable name.
    Example: "vision/noise_reducer.md" -> "vision_noise_reducer"
    """
    rel_path = os.path.relpath(path, root_dir)
    name = os.path.splitext(rel_path)[0]
    name = name.replace(os.sep, '_').replace('-', '_').replace(' ', '_')
    return ''.join(c if c.isalnum() or c == '_' else '_' for c in name)

def make_raw_string(html, delim='md'):
    return f'R"{delim}(\n{html}\n){delim}"'

def find_md_files_recursively(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.md'):
                yield os.path.join(dirpath, f)

def generate_header(input_dir, output_file):
    md_files = list(find_md_files_recursively(input_dir))

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("// Auto-generated header containing model card HTML strings\n")
        out.write("#pragma once\n\n")

        for md_path in md_files:
            with open(md_path, 'r', encoding='utf-8') as md_file:
                md_text = md_file.read()

            html = markdown.markdown(md_text)
            varname = sanitize_varname(md_path, input_dir)
            raw_html = make_raw_string(html)
            out.write(f'const char* {varname} = {raw_html};\n\n')

    print(f"Header written to: {output_file} ({len(md_files)} model cards found)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a C++ header from .md files recursively.")
    parser.add_argument("input_dir", help="Top-level directory to search for .md files")
    parser.add_argument("output_file", help="Output .h file")
    args = parser.parse_args()

    generate_header(args.input_dir, args.output_file)