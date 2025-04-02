import os

def generate_readme():
    root_dir = "."  # Current directory
    ignore_files = {".git", ".dvc", "__pycache__", ".github", "update_readme.py", "README.md"}
    
    def list_files(start_path, indent=0):
        result = ""
        for item in sorted(os.listdir(start_path)):
            if item in ignore_files:
                continue
            path = os.path.join(start_path, item)
            if os.path.isdir(path):
                result += " " * indent + f"- 📂 {item}/\n"
                result += list_files(path, indent + 2)
            else:
                result += " " * indent + f"- 📄 {item}\n"
        return result

    content = f"# MLOps Weather in Australia\n\n## Project Structure\n\n{list_files(root_dir)}"
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    generate_readme()

