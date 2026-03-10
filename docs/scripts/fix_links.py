import os
import re
from pathlib import Path

def fix_links(root_dir):
    """
    Recursively scans markdown files and converts absolute links (starting with /) 
    to relative links. Also handles the case where links point to .md files,
    converting them to relative paths that Sphinx can resolve (usually implies keeping .md for MyST,
    but we need to be careful about /docs/ prefix).
    """
    # Match [text](/path/to/file) or [text](/path/to/file.md)
    link_pattern = re.compile(r'\[([^\]]+)\]\((/[^)]+)\)')
    
    docs_root = Path(root_dir).resolve()

    for root, dirs, files in os.walk(root_dir):
        if "build" in root or "node_modules" in root or ".git" in root or "scripts" in root:
            continue
            
        for file in files:
            if not file.endswith(".md"):
                continue
                
            file_path = Path(root) / file
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            def replace_link(match):
                text = match.group(1)
                abs_path = match.group(2)
                
                # Logic to determine target path relative to docs root
                target_path_str = abs_path
                
                # Case 1: Starts with /docs/ -> Remove /docs/ to get path relative to docs root
                if target_path_str.startswith("/docs/"):
                    target_path_str = target_path_str[6:]
                # Case 2: Starts with / but not /docs/ -> Treat as relative to docs root (if it points to a file in docs)
                # But wait, if it points to /src/..., we should probably keep it relative to project root?
                # Sphinx usually builds from docs/source or docs/ root.
                # If we assume all doc links are internal to docs/, then /foo.md means docs/foo.md
                elif target_path_str.startswith("/"):
                    target_path_str = target_path_str[1:]
                
                # Check if it looks like an internal doc link
                # (e.g. guides/cli.md)
                
                try:
                    current_file_abs = file_path.resolve()
                    
                    # Construct target absolute path assuming it's in docs_root
                    target_abs = docs_root / target_path_str
                    
                    # Calculate relative path from current file's directory to target file
                    rel_path = os.path.relpath(target_abs, current_file_abs.parent)
                    
                    # Special handling: if target is outside docs root (e.g. ../src/...), keep it as is or fix?
                    # But here we are focusing on internal doc links.
                    
                    # If it was a /docs/ link, we definitely want to make it relative.
                    if abs_path.startswith("/docs/"):
                         return f"[{text}]({rel_path})"
                    
                    # If it was just /, and the file exists in docs, make it relative
                    if target_abs.exists() or target_abs.with_suffix(".md").exists():
                         return f"[{text}]({rel_path})"

                    # Fallback: if we can't verify existence, but it looks like a doc path
                    if "guides/" in abs_path or "reference/" in abs_path or "policy/" in abs_path:
                         return f"[{text}]({rel_path})"

                    return match.group(0)
                except Exception as e:
                    print(f"Error processing link {abs_path} in {file_path}: {e}")
                    return match.group(0)

            new_content = link_pattern.sub(replace_link, content)
            
            if new_content != content:
                print(f"Fixed links in {file_path}")
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

if __name__ == "__main__":
    fix_links(".")
