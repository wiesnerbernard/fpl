# Python Code Standards & Best Practices

This guide outlines the standards and best practices for Python code in this repository, with a focus on maintainability, readability, and suitability for AI/model development.

## 1. General Style
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for code style.
- Use 4 spaces per indentation level (no tabs).
- Limit lines to 79 characters.
- Use meaningful variable, function, and class names.

## 2. Imports
- Group imports: standard library, third-party, local modules.
- Use absolute imports where possible.
- Avoid wildcard imports (`from module import *`).

## 3. Type Annotations
- Use type hints for function arguments and return values.
- Example:
  ```python
  def predict(input_data: pd.DataFrame) -> np.ndarray:
      ...
  ```

## 4. Docstrings & Comments
- Use triple-quoted docstrings for all public modules, classes, and functions (PEP 257).
- Write clear, concise comments where necessary.
- Example:
  ```python
  def train_model(X: pd.DataFrame, y: pd.Series) -> Model:
      """Train a model on the given data and return the trained model."""
      ...
  ```

## 5. Code Organization
- Separate code into logical modules and functions.
- Avoid long functions (>40 lines).
- Place script entry points under `if __name__ == "__main__":`.

## 6. Error Handling
- Use exceptions for error handling, not return codes.
- Catch specific exceptions, not bare `except:`.

## 7. Data Handling
- Use pandas DataFrames for tabular data.
- Validate input data shapes and types.

## 8. Model Code
- Save models with versioning (e.g., using joblib, pickle, or ONNX).
- Document model input/output formats.
- Use configuration files (YAML/JSON) for hyperparameters.

## 9. Testing
- Write unit tests for all major functions and classes.
- Use `pytest` or `unittest`.
- Place tests in a `tests/` directory or files prefixed with `test_`.

## 10. Dependencies
- List all dependencies in `requirements.txt` or `pyproject.toml`.
- Pin versions for reproducibility.

## 11. Linting & Formatting
- Use tools like `flake8`, `black`, or `isort` for linting and formatting.
- Run linters before committing code.

## 12. Notebooks
- Keep notebooks clean: remove unnecessary outputs and keep code cells concise.
- Move reusable code to `.py` modules.

---

For more details, see:
- [PEP 8 – Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
- [PEP 257 – Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
