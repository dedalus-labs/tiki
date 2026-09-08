# Contributing to Tiki

We want to make contributing to this project as easy and transparent as
possible.

## AI assistance

We encourage AI assistance with code, tests, documentation, and communication.
You are responsible for understanding, reviewing, and verifying everything you
submit. AI-assisted work must follow the same code, style, and communication
standards as any other contribution.

Your chances of getting a pull request merged are higher when its explanation
is written in your own words and shows that you understand the change.
This is a preference, not a ban on AI-written descriptions. No authorship
attestation is required. Mention AI assistance when it helps reviewers assess
the work, and never claim human authorship or verification that did not occur.

## Code and communication

- Follow the conventions of the surrounding code and the repository formatters.
- Keep changes focused. Add tests for changed behavior and document public APIs.
- Explain why a comment is needed. Keep comments concise and avoid repeating code.
- Write issues, pull requests, commit messages, and review replies in plain English.
- Describe the problem, resulting behavior, and evidence. Report failures and
  untested cases explicitly. Do not paste unreviewed model output or chat transcripts.
- Preserve upstream license and attribution notices. Changes submitted to MLX
  upstream must follow that project's contribution policy.

## Commits

Use a short, imperative title in the form `type(scope): description`, such as
`fix(cuda): retain buffers until collective completion`. Use `feat`, `fix`,
`refactor`, `docs`, `test`, or `chore` as appropriate. Keep the title under 72
characters and each commit focused on one change. Add a body only when the
reason, compatibility impact, or validation needs explanation.

An optional local commit template is available:

```sh
git config --local commit.template .github/commit_template.txt
```

## Pull Requests

- Make sure new code is covered by tests. Add new tests if not, and confirm
  the new tests fail in the main branch.
- If performance may be impacted, run benchmarks for both the main branch and
  the pull request.
- When providing benchmarking results, include scripts and reproduction steps.
- Format the code with `uvx pre-commit run --all` before submitting a pull
  request. You can also install git hooks to run it automatically:

  ```shell
  pip install pre-commit
  pre-commit install
  ```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to Tiki, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
