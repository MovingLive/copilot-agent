# Custom Instructions for project MovingLive

Every time you choose to apply a rule(s), explicitly state the rule(s) in the output. You can abbreviate the rule description to a single word or phrase.

## Project Context

GitHub Copilot Extensions who are using RAG (FAISS embeddings) and Copilot LLM to a assist developers in writing code.

## Code Style and Structure

- Write concise, technical TypeScript and Python code with accurate examples
- Use functional and declarative programming patterns; avoid classes
- Prefer iteration and modularization over code duplication
- Use descriptive variable names with auxiliary verbs (e.g., isLoading, hasError)
- Structure repository files as follows:

```bash
Workspace
├── .github # Contient les configurations GitHub
│ ├── workflows # Workflows CI/CD
│ ├── copilot-instructions.md
├── app # Contient le code source du backend
│ ├── main.py # Point d'entrée de l'application
├── logs # Fichiers de journalisation
├── tests # Tests unitaires et d'intégration
├── .env.development # Variables d'environnement de développement
├── .env.production # Variables d'environnement de production
├── pyproject.toml # Dépendances Python avec Poetry
├── pytest.ini # Configuration des tests
└── README.md # Documentation du projet
```

## Tech Stack

- Python 3.12
- Poetry
- Fast API
- Pydantic V2
- PyUnit
- GitHub Action

## Naming Conventions

- Use lowercase with dashes for directories (e.g., components/form-wizard)
- Use snake_case for functions and variables
- Favor named exports for components and utilities
- Use PascalCase for component files (e.g., VisaForm.tsx)
- Use camelCase for utility files (e.g., formValidator.ts)

## Python Usage

- Use `now(timezone.utc)` instead of `utcnow` method
- Use lazy % formatting in logging functions
- Consider explicitly re-raising using 'raise HTTPException(status_code=401, detail='Invalid GitHub token') from e'

## FastAPI Usage

- Use lifespan event handlers instead of on_event `Method`

## Pydantic Usage

- Use `model_config = ConfigDict(from_attributes=True)` for mapping object to database
- Use `@field_validator` and `@classmethod` for personalized validation

Exemple:

```python
from pydantic import BaseModel, ConfigDict, field_validator
@field_validator("name", mode="before")
@classmethod
def validate_name(cls, v):
if not v:
raise ValueError("name cannot be empty")
return v
```

## Syntax and Formatting

### Back-end: Python

- Variables and Functions: Use snake_case, which means lowercase letters with underscores separating words. For example, my_variable or my_function().
- Constants: Use UPPER_CASE_WITH_UNDERSCORES for constants. For example, MY_CONSTANT.
- Classes: Use PascalCase, where each word starts with an uppercase letter without separators. For example, MyClass.

## Error Handling

- Implement proper error boundaries
- Log errors appropriately for debugging
- Provide user-friendly error messages
- Handle network failures gracefully

## Testing

- Write unit tests for utilities and components
- Implement E2E tests for critical flows
- Test across different Chrome versions
- Test memory usage and performance

## Security

- Implement Content Security Policy
- Sanitize user inputs
- Handle sensitive data properly
- Implement proper CORS handling
