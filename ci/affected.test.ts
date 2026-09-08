import assert from "node:assert/strict";
import { mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { nodeExec, type ScriptExec } from "@dedalus-labs/hollywood";
import { affectedForEvent, classify } from "./affected.ts";

test("policy and prose changes do not select native builds", () => {
  assert.deepEqual(
    classify([
      "AGENTS.md",
      "CONTRIBUTING.md",
      ".github/rulesets/main.json",
      ".github/rulesets/README.md",
      ".github/CODEOWNERS",
      ".github/commit_template.txt",
      ".github/ISSUE_TEMPLATE/bug_report.md",
      ".github/pull_request_template.md",
      ".github/workflows/build_and_test.yml",
    ]),
    { native: false, lint: false, dependencies: false },
  );
});

test("native sources, build inputs, and unknown paths select native builds", () => {
  for (const file of [
    "mlx/array.cpp",
    "mlx/backend/cuda/runtime/src/allocation.rs",
    "tests/test.cpp",
    "CMakeLists.txt",
    "cmake/FindNCCL.cmake",
    "python/src/array.cpp",
    "python/tests/test_ops.py",
    "ci/native.ts",
    ".github/actions/build/action.yml",
    ".github/scripts/build-sanitizer-tests.sh",
    "new-subsystem/config.json",
  ]) {
    assert.equal(classify([file]).native, true, file);
  }
});

test("dependency and formatting checks select their actual inputs", () => {
  for (const file of [
    "package-lock.json",
    "Cargo.lock",
    "mlx/backend/cuda/runtime/Cargo.toml",
    "docs/requirements.txt",
    "pyproject.toml",
    "setup.py",
  ]) {
    assert.equal(classify([file]).dependencies, true, file);
  }
  for (const file of [
    "examples/scan.py",
    "mlx/array.cpp",
    ".pre-commit-config.yaml",
    "cmake/flags.cmake",
  ]) {
    assert.equal(classify([file]).lint, true, file);
  }
});

test("failed or incomplete diffs cannot classify a change as irrelevant", async () => {
  const exec: ScriptExec = async () => {
    throw new Error("missing Git history");
  };
  await assert.rejects(
    affectedForEvent(exec, { event: "pull_request", base: "a".repeat(40), head: "b".repeat(40) }),
    /missing Git history/,
  );
  await assert.rejects(
    affectedForEvent(exec, { event: "merge_group", base: "", head: "b".repeat(40) }),
    /full commit SHA/,
  );
  assert.deepEqual(
    await affectedForEvent(exec, { event: "workflow_dispatch", base: "", head: "" }),
    { native: true, lint: true, dependencies: true },
  );
  assert.deepEqual(
    await affectedForEvent(exec, { event: "push", base: "0".repeat(40), head: "b".repeat(40) }),
    { native: true, lint: true, dependencies: true },
  );
});

test("real Git diffs retain deleted and renamed native paths and arbitrary filenames", async () => {
  const root = await mkdtemp(join(tmpdir(), "tiki-ci-diff-"));
  const exec: ScriptExec = (file, args, options) => nodeExec(file, args, { ...options, cwd: root });
  const git = async (...args: string[]) => (await exec("git", args)).stdout.trim();
  try {
    await git("init", "-b", "main");
    await git("config", "user.name", "CI fixture");
    await git("config", "user.email", "ci@example.invalid");
    await writeFile(join(root, "kernel.cpp"), "int value = 1;\n");
    await git("add", "kernel.cpp");
    await git("commit", "-m", "base");
    const base = await git("rev-parse", "HEAD");
    await git("switch", "-c", "proposal");
    await rename(join(root, "kernel.cpp"), join(root, "kernel.md"));
    await writeFile(join(root, "weird\nname.cpp"), "int value = 2;\n");
    await git("add", "--all");
    await git("commit", "-m", "rename");
    const head = await git("rev-parse", "HEAD");
    for (const event of ["pull_request", "merge_group", "push"] as const) {
      assert.equal((await affectedForEvent(exec, { event, base, head })).native, true);
    }
    await git("switch", "main");
    await writeFile(join(root, "main-only.cpp"), "int value = 3;\n");
    await git("add", "main-only.cpp");
    await git("commit", "-m", "advance base");
    await git("switch", "-c", "documentation", base);
    await writeFile(join(root, "README.md"), "Docs\n");
    await git("add", "README.md");
    await git("commit", "-m", "docs");
    assert.equal(
      (
        await affectedForEvent(exec, {
          event: "pull_request",
          base: await git("rev-parse", "main"),
          head: await git("rev-parse", "HEAD"),
        })
      ).native,
      false,
    );
  } finally {
    await rm(root, { recursive: true });
  }
});
