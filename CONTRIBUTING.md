# Contributing to MedXplain-VQA

We welcome contributions from the community! This document provides guidelines for contributing to the MedXplain-VQA project.

## 🤝 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue templates** when creating new issues
3. **Provide clear reproduction steps** for bugs
4. **Include system information** (OS, Python version, etc.)

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding standards** outlined below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure CI passes** before submitting

## 🛠️ Development Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU support)

### Local Development

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/medxplain-vqa.git
cd medxplain-vqa

# Create development environment
conda create -n medxplain-dev python=3.8
conda activate medxplain-dev

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
python -m pytest test/

# Run with coverage
python -m pytest test/ --cov=src/

# Run specific test file
python -m pytest test/test_model.py
```

## 📝 Coding Standards

### Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **mypy**: Type checking

```bash
# Format code
black src/ scripts/ test/

# Check linting
flake8 src/ scripts/ test/

# Sort imports
isort src/ scripts/ test/

# Type checking
mypy src/
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add new explainability component
fix: resolve memory leak in grad-cam
docs: update installation instructions
test: add unit tests for medical evaluation
```

### Python Code Guidelines

- Use **type hints** for all function signatures
- Add **docstrings** for all public functions and classes
- Follow **PEP 8** style guidelines
- Keep functions **focused and small**
- Use **meaningful variable names**

Example:

```python
def process_medical_image(
    image: PIL.Image.Image, 
    question: str,
    model_config: Dict[str, Any]
) -> MedicalVQAResult:
    """
    Process a medical image with question to generate explainable answer.
    
    Args:
        image: Input medical image
        question: Natural language question
        model_config: Model configuration parameters
        
    Returns:
        MedicalVQAResult containing answer and explanations
        
    Raises:
        ValueError: If image format is not supported
    """
    # Implementation here
    pass
```

## 🔬 Research Contributions

### Adding New Components

When adding new explainability components:

1. **Create component in appropriate module** (`src/explainability/`)
2. **Add comprehensive tests** (`test/explainability/`)
3. **Update configuration** (`configs/config.yaml`)
4. **Add evaluation metrics** if needed
5. **Update documentation** and README

### Evaluation and Benchmarks

When adding new evaluation methods:

1. **Follow medical-domain standards**
2. **Include statistical significance testing**
3. **Provide comparison with baselines**
4. **Add visualization code**
5. **Document methodology clearly**

## 📊 Data and Experiments

### Adding New Datasets

1. **Create data loader** in `src/preprocessing/`
2. **Add dataset configuration** 
3. **Include data validation**
4. **Update evaluation scripts**
5. **Provide dataset documentation**

### Experiment Reproducibility

- **Pin all dependency versions**
- **Set random seeds consistently**
- **Document hardware requirements**
- **Provide configuration files**
- **Include result validation**

## 📚 Documentation

### Code Documentation

- **Module-level docstrings** explaining purpose
- **Class docstrings** with usage examples
- **Function docstrings** with Args/Returns/Raises
- **Inline comments** for complex logic

### User Documentation

- **Update README.md** for new features
- **Add usage examples** in `examples/`
- **Create tutorials** for complex workflows
- **Update API documentation**

## 🧪 Testing Guidelines

### Test Coverage

- **Unit tests** for individual components
- **Integration tests** for end-to-end workflows
- **Medical evaluation tests** for domain-specific metrics
- **Performance tests** for computational efficiency

### Test Organization

```
test/
├── unit/                   # Unit tests
│   ├── test_models.py
│   ├── test_explainability.py
│   └── test_utils.py
├── integration/           # Integration tests
│   ├── test_pipeline.py
│   └── test_evaluation.py
├── medical/              # Medical domain tests
│   ├── test_medical_metrics.py
│   └── test_clinical_validation.py
└── fixtures/             # Test data and fixtures
    ├── sample_images/
    └── sample_questions.json
```

## 🏷️ Release Process

### Version Numbers

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git

## 🌟 Recognition

Contributors will be acknowledged in:

- **README.md** contributors section
- **Paper acknowledgments** (for significant contributions)
- **Release notes** for major features
- **Project documentation**

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Email**: Contact maintainers directly for sensitive issues

## 📋 Issue Templates

### Bug Report Template

```markdown
**Describe the bug**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Configure with '...'
2. Run command '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.8.10]
- PyTorch version: [e.g. 2.1.0]
- CUDA version: [e.g. 11.8]

**Additional context**
Any other context about the problem.
```

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Other solutions you've considered.

**Additional context**
Any other context about the feature request.
```

## 🎯 Priorities

Current development priorities:

1. **Medical Expert Validation**: Integration with clinical workflows
2. **Performance Optimization**: Reducing inference time
3. **Dataset Expansion**: Support for additional medical imaging modalities
4. **Clinical Integration**: EHR and PACS integration
5. **Federated Learning**: Privacy-preserving training

Thank you for contributing to MedXplain-VQA! 🚀 