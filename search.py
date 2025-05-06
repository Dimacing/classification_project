import os

root_dir = "."
extensions = [".py", ".js", ".html", ".css"]
output_file = "output.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for root, dirs, files in os.walk(root_dir):
        # Исключаем папку .venv
        if ".venv" in dirs:
            dirs.remove(".venv")
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(f"=== {file_path} ===\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except Exception as e:
                    print(f"Ошибка чтения {file_path}: {e}")