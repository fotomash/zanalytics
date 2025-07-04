import os

MAIN_FILE = "🏠 Home.py"  # Change if your main file has a different name

def should_comment(path):
    return os.path.basename(path) != MAIN_FILE

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            new_lines = []
            changed = False
            for line in lines:
                # Only comment out top-level (not already commented) page config calls
#                 if "st.set_page_config" in line and not line.strip().startswith("#") and should_comment(path):
                    new_lines.append("# " + line)
                    changed = True
                else:
                    new_lines.append(line)
            if changed:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(new_lines)
#                 print(f"Commented out st.set_page_config in {path}")

# print("✅ Done. All st.set_page_config lines are now commented out everywhere except your main page.")