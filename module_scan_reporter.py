import os
import re

# Configuration
dir_root = './zanalytics_workspace'
orchestrator_files = [
    os.path.join(dir_root, 'advanced_smc_orchestrator.py'),
    os.path.join(dir_root, 'copilot_orchestrator.py')
]

# 1. Collect all .py files in the workspace
def collect_py_files(root):
    py_files = []
    for root_dir, _, files in os.walk(root):
        for f in files:
            if f.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root_dir, f), dir_root)
                py_files.append(rel_path.replace('\\', '/'))
    return set(py_files)

# 2. Parse imports from orchestrator files
def parse_imports(file_path):
    imports = set()
    pattern = re.compile(r"^(?:from|import)\s+([\w\.]+)")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    module = match.group(1)
                    imports.add(module)
    except FileNotFoundError:
        print(f"Warning: Orchestrator file not found: {file_path}")
    return imports

# 3. Map module names to expected file paths
# Converts module name (e.g., 'core.confirmation_engine_smc') to a path

def module_to_path(module_name):
    parts = module_name.split('.')
    return '/'.join(parts) + '.py'

# Run scan
all_py_files = collect_py_files(dir_root)
imported_modules = set()
for orch in orchestrator_files:
    imported_modules |= parse_imports(orch)
imported_paths = {module_to_path(m) for m in imported_modules}

# 4. Compare
missing = imported_paths - all_py_files
unused = all_py_files - imported_paths

# 5. Report
def report():
    print("--- Module Scan Report ---")
    print(f"Total Python files found: {len(all_py_files)}")
    print(f"Total modules imported by orchestrators: {len(imported_modules)}")
    print("\nMissing modules (imported but not found as files):")
    for m in sorted(missing):
        print(f"  - {m}")
    print("\nUnreferenced Python files (not imported by orchestrators):")
    for f in sorted(unused):
        print(f"  - {f}")
    print("\nChecklist: Ensure core, utils, wyckoff, and other directories are complete.")

if __name__ == '__main__':
    report()
