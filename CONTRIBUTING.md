# Contributing to PyParaglide

Thank you for your interest in contributing to PyParaglide!

## Development Setup

```bash
# Clone repository
git clone https://github.com/Genajoin/PyParaglide.git
cd PyParaglide

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_layers.py -v

# Run with coverage
pytest tests/ --cov=src/pyparaglide --cov-report=html

# Run fast tests only (skip integration tests)
pytest tests/ -m "not integration"
```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting and import sorting
- **MyPy** for type checking (gradual typing)

```bash
# Format code
black src/ tests/
ruff format src/ tests/

# Check linting
ruff check src/ tests/

# Type check
mypy src/
```

## Project Structure

When adding new features, follow this structure:

```
src/pyparaglide/
├── module_name/
│   ├── __init__.py      # Public API exports
│   ├── core.py          # Main implementation
│   └── utils.py         # Helper functions
└── tests/
    └── test_module.py   # Unit tests
```

## Adding New Features

### 1. Add CLI Command

Edit `src/pyparaglide/cli/__init__.py`:

```python
@app.command()
def my_feature(
    input_file: Path = typer.Option(..., help="Input file"),
    output: Path = typer.Option(..., help="Output file"),
):
    """My feature description."""
    console.print(f"Processing {input_file}...")
    # Your code here
```

### 2. Add New Layer

Edit `src/pyparaglide/models/layers.py`:

```python
class MyLayer(tf.keras.Model):
    """Layer description."""

    def __init__(self, param: int, name: str = "my_layer"):
        super().__init__(name=name)
        self.param = param
        # Define layers

    def build(self, input_shape):
        # Create weights
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Forward pass
        return outputs
```

### 3. Add Tests

Create `tests/test_my_feature.py`:

```python
import pytest
import numpy as np
import tensorflow as tf

from pyparaglide.my_module import my_function


class TestMyFeature:
    def test_basic(self):
        result = my_function(input_data)
        assert result.shape == expected_shape
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Format code (`black src/ tests/ && ruff check src/ tests/`)
7. Commit your changes (`git commit -m "Add amazing feature"`)
8. Push to branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Commit Message Format

We follow semantic commit messages:

```
feat: add new amazing feature
fix: correct bug in wind processing
docs: update API documentation
test: add tests for normalization
refactor: simplify data loading
perf: improve training speed
```

## Questions?

- Open an issue for bugs or feature requests
- Check existing [issues](https://github.com/Genajoin/PyParaglide/issues)
- See [Architecture](docs/ARCHITECTURE.md) for technical details
