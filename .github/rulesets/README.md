# Main branch protection

`main.json` is the GitHub ruleset payload for `dedalus-labs/tiki`.
Committing this file does not apply settings. A repository administrator must
apply it through the rulesets API or import it in GitHub Settings.

The policy follows [BSMR's main ruleset](https://github.com/dedalus-labs/bsmr/rules/20017484):
one approving review, code-owner review where ownership is declared, linear
history, squash-only pull requests, a merge queue, and no branch deletion or
force-pushes. Organization administrators retain BSMR's explicit bypass.
Stale approvals are not dismissed, and thread resolution and approval of the
last push are not required, matching BSMR.

Required checks use Tiki's existing lint, ASAN, UBSAN, and Fedora builds in place
of BSMR's generated-workflow and Rust checks. Dependency review uses BSMR's high
severity threshold. Each required check runs for pull requests and merge groups.
The upstream-only MLX build jobs are not required because they skip on Tiki.
These checks do not certify Tiki's Rust CUDA runtime or distributed training.

Before enabling the queue, ensure the candidate pull request includes the
`merge_group` workflow trigger. Older open pull requests must incorporate this
change before entering the queue. CODEOWNERS takes effect after it reaches main.

For first-time installation:

```sh
gh api --method POST repos/dedalus-labs/tiki/rulesets --input .github/rulesets/main.json
```

For subsequent changes, inspect `gh api repos/dedalus-labs/tiki/rulesets`, find
the existing main ruleset ID, and PUT the payload to that ruleset's endpoint.
Read the ruleset back after applying it and compare its rules, conditions,
enforcement, and bypass actors with this file. Do not create duplicate rulesets.
