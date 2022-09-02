# Engineering Processes

Here are a couple things to keep in mind when submitting a pull request to this repo. We're
excited to see what you have in mind :relaxed:

Note that many of these processes are in place due to the complexity of working in Deep
Learning! A challenge that we embrace.

## Exploration

Before starting development on a feature, we take the time to explore different approaches. To
assist with that, we follow the
[The Double Diamond Model](https://towardsdatascience.com/structure-your-data-science-project-the-double-diamond-model-3dfbf64e129a)
. Please challenge yourself to think laterally and be open-minded to crazy
:stuck_out_tongue_winking_eye: ideas as those lead to innovation.

To help us all think out side the box, when you can, please document the alternative solutions
you considered in your pull requests!

## Development

### Documentation

Let's write simple, FUN, and readable documentation. Reading our documentation should be like
breathing fresh air, refreshing.

To help facilitate that, we ask that you: avoid writing jargon, document edge cases, write
documentation that lasts, and offer examples. Last but not least, take the time to get feedback
from your team mates to ensure others, especially more junior or less technical team members, can
understand it. We aim to write inviting and inclusive documentation.

If you don't understand parts of our documentation, please let us know!

#### Learnings & Assumptions

Our work tends to span multiple disciplines including audio engineering, linguistics, and deep
learning. For us to stay aligned, it's important that we document our learnings and assumptions.
Here are a couple of ways in which we have done that:

- We describe the thought process behind our complex implementations.
- We abstract the code such that all our important decisions and are easy to find. For
example, many of our magic numbers are defined in one module, `_config`.
- We quote from or link to original sources often.

Help us all stay on the same page by documenting your learnings and assumptions!

### Testing

Where possible, we prefer that code is tested. For Python, we use `pytest` for testing like so:

```bash
python -m pytest
```

After the tests run, we use `pytest-cov` to analyze and print the test case code coverage. Whenever
possible, we prefer to increase the coverage as an indication that more of our software is tested.
With that said, we don't over optimize for code coverage.

### Coding Style

To aid code readability, we prefer to use Google coding style. Please familiarize yourself Google's
Python style guide [here](http://google.github.io/styleguide/pyguide.html).

Where possible, we use auto-formatters and linters to enforce code readability. For Python,
we use `black`, `isort`, `pylance`, and `flake8`. These tools should be readily available through
your code editor.

Last but not least, we use tools like `markdownlint` and "Spell Right", to assist us in writing
and formatting our documentation.

### Technical Debt

To help ensure that our software is inviting and inclusive, it's important that we limit any debt
that could obscure our data processing or modeling. Our processing layers should be intuitive
and holistic.

## Legal

### Dependencies

To ensure we only use code we are allowed to use, every so often we double check the licenses of the
installed packages, like so:

`python -m run.utils.check_licenses`

Before adding additional dependencies, please check if the license is standard and permissible.

### Security

Our users submit sensitive data to our service all the time. Please familiarize yourself with our
security protocols so that we are able to maintain trust with our users and can continue passing our
security audits.

## Deployment

### Releases

Every quarter or so we aim to release a major upgrade to our text-to-speech model. Our release
cycle is slow to accommodate a comprehensive quality assurance process and to limit disruption for
our customers. Each of our models has a unique sound which makes product consistency challenging.

More often, we'll make minor changes. In the past this has included updates to our text processing,
error handling, voice library, and/or experimental models.

### Bugs

To track bugs, we write inline comments, write test cases, and/or use an external tool like Jira.
To fix bugs, we ask that you please add regression tests, so that we don't run into same issue
twice.
