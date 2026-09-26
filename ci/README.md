# GitHub Actions

All four workflow definitions are authored in TypeScript with Hollywood 0.0.5.
The version and its dependencies are pinned in `package-lock.json`.
Edit `ci/*.ts`, then regenerate the committed workflow YAML and action bundles:

```sh
npm ci --ignore-scripts
npm run ci generate
npm run ci check
```

The check typechecks the source, runs invariant tests, regenerates all workflows
and local actions, and rejects stale or missing committed output. `command`
represents one executable and its arguments. The local actions use typed inputs
and `exec` for branching, loops, and process execution. Do not use `unsafeShell`
or embed scripts in workflow definitions.

## Path selection

CI uses the complete local Git diff, without the GitHub changed-file API limit.
Pull requests use the merge base, merge groups use their supplied base and head,
and pushes compare their before and after commits. Manual runs and new-branch
pushes select every lane. Invalid SHAs or missing Git history fail the workflow.
Renames include both the old and new path so moving native code into a docs
directory cannot hide a deletion.

Policy, templates, and prose do not select native builds. Native sources, tests,
build inputs, native action changes, and unknown paths do. Python and C++ files
select formatting checks. Dependency manifests select dependency review on PRs
and merge groups. Workflow source and generated-file validation runs on every
event, including changes to the path-selection policy itself.

`Check Lint` retains its existing required-check name and now validates the
complete CI result. It accepts a skipped job only when path detection succeeded
and marked that lane irrelevant. A failed, cancelled, or unexpectedly skipped
selected job fails this check. No branch-settings update is required.

## Runner approval

Selected native builds wait on the `dedalus-machines-ci` GitHub environment before GitHub
assigns a runner. Its sole required reviewer is `@windsornguyen`. Self-approval
is allowed so the maintainer can approve runs on their own PRs. Environment
bypass is disabled. The main-branch administrator merge bypass is unchanged.

Each new workflow run, including a new PR commit or merge-group revision,
requires approval through **Review deployments** in the Actions run. Rejecting
or cancelling native work cannot make the required result check pass. Changes
that do not select native builds do not request approval. Cheap path detection,
workflow validation, formatting, and dependency review can run first.

GitHub also requires workflow approval for every external contributor. That
repository setting is independent of the maintainer-only native build gate.

Dedalus Machines routing is not active yet. Linux x64 capacity supports repository-scoped,
one-job registrations. It runs unprivileged containers with a read-only root
and no Docker socket. Tiki's existing native actions require package installation
or nested Docker, so their runner assignments remain behind approval until a
prepared build image and per-run registration owner are integrated. No coordinator
credential belongs in this public repository.

The requested provider order is Dedalus Machines, Blacksmith, then GitHub-hosted. Follow the
Dedalus ownership rule: only unavailable capacity before workload assignment can
advance to another provider. Confirm cancellation and resource cleanup first.
Once execution starts, a test failure, timeout, or out-of-memory error remains a
failure. Preserve OS, architecture, tool availability, and memory requirements
across every eligible route. Registration alone does not activate this policy.

## Migration scope

The active sanitizer and Fedora jobs now execute through Hollywood actions.
The existing nine platform-specific composite actions remain dependencies of
the inherited MLX build, documentation, and release workflows. Their shell
programs have not been ported in this change. Upstream-only build guards and
PyPI targets remain explicit. The interaction-bypass workflow runs only in MLX
upstream because its credential and API target belong to that repository.

ARM builds retain their tested GitHub-hosted runners. The x86 Fedora build uses
Blacksmith. This workflow migration does not qualify CUDA or distributed
training, and it does not publish a package or deploy documentation during PR CI.
