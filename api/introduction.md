site_name: ZSI Copilot Documentation
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Architecture: architecture.md
  - Agents: agents.md
  - Flows: flows.md
  - Schemas: schemas.md
  - System: system.md
  - Persistence: persistence.md
  - Security: security.md
theme:
  name: material
  palette:
    primary: ${primary_color}
    accent: ${accent_color}
plugins:
  - search
  - mkdocs-techdocs-core
  - environment
extra:
  repo_url: https://github.com/yourusername/zsi-copilot
  edit_uri: edit/main/docs/
environment:
  file: mkdocs/env.yml
  vars:
    primary_color: indigo
    accent_color: indigo
