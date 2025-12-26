# AGENTS.md
 
## Dev environment tips
- Use `uv` tool and package manager
- Use `uv tool run ruff format blokus_env` & `uv tool run ruff format tests` for linting and formatting
- Use `uv tool run mypy blokus_env` for type checking
- You are a Test Driven Developer. Making sure that every change is well tested with unit and integration tests.
- Use `uv tool run pytest tests` to execute the tests.
 
## Coding guidelines
- Use short functions with descriptive names
- Use comments sparingly. Only when they add value and explain something that is not clear from a function call or a variable name.
