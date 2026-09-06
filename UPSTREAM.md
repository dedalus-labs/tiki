# Upstream policy

Tiki retains the complete MLX history. The `upstream` remote points to
`ml-explore/mlx`, and `upstream-main` is an exact mirror of its `main` branch.

Import upstream changes with a merge commit:

```sh
git fetch upstream
git switch upstream-main
git merge --ff-only upstream/main
git push origin upstream-main
git switch main
git merge --no-ff upstream-main
```

Do not commit Tiki changes to `upstream-main`. Keep framework changes localized
so `git diff upstream-main...main` remains a useful description of Tiki.

Use build configuration to exclude unwanted components. Delete upstream source
only when the deletion removes an active maintenance or correctness burden.
