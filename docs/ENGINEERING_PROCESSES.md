# Engineering Processes

## Testing

Where possible, we prefer that code is tested. For Python, we use `pytest` for testing like so:

```bash
pytest
```

After the tests run, we use `pytest-cov` to analyze and print the test case code coverage. Whenever
possible, we prefer to increase the coverage as an indication that more of our software is tested.

## Coding Style

To aid code readability, we prefer to use Google coding style. Please familiarize yourself Google's
Python style guide [here](http://google.github.io/styleguide/pyguide.html).

Where possible, we also use auto-formatters and linters to enforce code readability. For Python,
we use `yapf` and `flake8`. These tools can be used like so:

```bash
# Run our linter, learn more: https://en.wikipedia.org/wiki/Lint_(software)
flake8 src/; flake8 tests/;

# Automatically format your code.
yapf src/ tests/ --recursive --in-place;
```

You'll want to integrate `yapf` and `flake8` into your code editor, both of these tools are
popular and there are many extensions written for them.

## Reporting Bugs

First, do [a quick search](https://github.com/wellsaid-labs/Text-to-Speech/issues) to see whether
your issue has already been reported. If your issue has already been reported, please comment on
the existing issue.

Otherwise, open [a new GitHub issue](https://github.com/wellsaid-labs/Text-to-Speech/issues).
