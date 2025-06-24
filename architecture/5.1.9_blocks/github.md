# **/.github/workflows/ \- GitHub Actions CI/CD Workflows**

This directory contains YAML files that define GitHub Actions workflows for Continuous Integration (CI) and potentially Continuous Deployment (CD) for the Zanzibar Analytics project.

## **Purpose**

* **Automated Quality Assurance:** Automatically run checks (like linting and testing) on every push or pull request to key branches (e.g., main, dev).  
* **Early Bug Detection:** Catch integration issues, regressions, and coding standard violations early in the development cycle.  
* **Consistency:** Ensure that all code merged into primary branches meets a defined quality standard.  
* **Deployment Automation (Future):** Can be extended to automate build, packaging, and deployment processes.

## **Key Files**

* **python-ci.yml (Version: 1.0)**  
  * **Trigger:** Runs on pushes and pull requests to the main and dev branches. Also allows manual triggering via workflow\_dispatch.  
  * **Environment:** Executes on an ubuntu-latest runner.  
  * **Python Versions:** Uses a matrix strategy to test against multiple Python versions (e.g., 3.9, 3.10, 3.11) to ensure compatibility.  
  * **Steps:**  
    1. **Checkout Repository:** Fetches the latest code.  
    2. **Set up Python:** Installs the specified Python version from the matrix. Caches pip dependencies to speed up subsequent runs.  
    3. **Install Dependencies:** Installs the project in editable mode (pip install \-e .\[dev\]) along with development dependencies specified in pyproject.toml.  
    4. **Lint with Ruff:** Runs ruff check . to identify linting issues and ruff format \--check . to verify code formatting against the rules defined in pyproject.toml.  
    5. **Test with pytest:** Executes the project's test suite using pytest. pytest will use its configuration from pyproject.toml (e.g., for test discovery paths and coverage).  
    6. **(Optional Placeholder) Upload Coverage:** Includes a commented-out step for potentially uploading test coverage reports to a service like Codecov.

## **Management**

* Workflow files are version-controlled as part of the repository.  
* **Secrets:** Any sensitive information required by workflows (e.g., API tokens for deployment or services like Codecov) should be stored as encrypted secrets in the GitHub repository settings (Settings \-\> Secrets and variables \-\> Actions) and accessed in the workflow using ${{ secrets.YOUR\_SECRET\_NAME }}.  
* **Workflow Status:** The status of workflow runs can be monitored from the "Actions" tab of the GitHub repository.

## **Future Enhancements**

* Add steps for building documentation (e.g., using Sphinx).  
* Implement steps for building and publishing the Python package (e.g., to PyPI or a private repository).  
* Add deployment workflows for different environments (staging, production).  
* Integrate more advanced static analysis tools or security scanners.