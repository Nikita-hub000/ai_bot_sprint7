import os
import re
import ast

SOURCE_DIR = "source"
OUTPUT_DIR = "knowledge_base"
TERMS_FILE = "terms_map.txt"


def load_terms_map(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    dict_text = "{" + content.rstrip(",\n ") + "}"
    return ast.literal_eval(dict_text)


def replace_terms(text, terms_map):
    for old, new in terms_map.items():
        pattern = r'\b{}\b'.format(re.escape(old))
        text = re.sub(pattern, new, text)
    return text


def process_files():
    terms_map = load_terms_map(TERMS_FILE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".txt"):
            source_path = os.path.join(SOURCE_DIR, filename)

            with open(source_path, "r", encoding="utf-8") as f:
                content = f.read()

            updated_content = replace_terms(content, terms_map)

            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_kota{ext}"
            output_path = os.path.join(OUTPUT_DIR, new_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print(f"Обработан: {filename} → {new_filename}")


if __name__ == "__main__":
    process_files()