# Contributing to Opfunu

We appreciate your interest in contributing to Opfunu! This guide details how to contribute in a way that is efficient 
for everyone.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Pull Requests](#pull-requests)
- [Issue Tracking](#issue-tracking)

## Code of Conduct

All contributors are expected to adhere to the project's [Code of Conduct](CODE_OF_CONDUCT.md). Please read the document before contributing. 

## Getting Started

1. Fork the project repository and clone your fork:

    ```
    git clone https://github.com/thieu1995/opfunu.git
    ```

2. Create a new branch for your changes:

    ```
    git checkout -b name-of-your-branch
    ```

3. Make your changes and commit them:

    ```
    git commit -m "Detailed commit message"
    ```

4. Push your changes to your fork:

    ```
    git push origin name-of-your-branch
    ```

5. Create a pull request from your branch to the Opfunu main branch.

## How Can I Contribute?

Here are some ways to contribute:

- Improve documentation
- Fix bugs or add new features
- Write tutorials or blog posts
- Review code submissions
- Test the application and report issues

However, before contributing, make sure that the unit tests pass and that new functionality is covered by unit tests. 
The unit tests can be run using `pytest`. Change working directory to opfunu and then use:

```python
# Test CEC-based functions
python -m pytest tests/cec_based

# Test Name-based functions
python -m pytest tests/name_based
```

Or you can test all files by:

```python
python -m pytest
```


## Pull Requests

[Pull requests](https://github.com/thieu1995/opfunu/pulls) are the best way to propose changes to the codebase. We 
actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Issue that pull request!

## Issue Tracking

We use [GitHub issues](https://github.com/thieu1995/opfunu/issues) to track public bugs and requests. Please ensure 
your description is clear and has sufficient instructions to be able to reproduce the issue.

## Any questions?

Don't hesitate to contact us if you have any questions. Contact [@thieu1995](mailto:nguyenthieu2102@gmail.com)
or ask your question on issues.

Thank you for your contributions!
