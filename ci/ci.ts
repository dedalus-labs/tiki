import { command, job, uses, workflow, type GitHubStepWorkflowJob } from "@dedalus-labs/hollywood";
import {
  always,
  and,
  eq,
  expr,
  format,
  github,
  needsOutput,
  needsResult,
  not,
  or,
  stepOutput,
} from "@dedalus-labs/hollywood/expr";

import { affected } from "./affected.ts";
import { native } from "./native.ts";
import { checkout, install, node } from "./steps.ts";
import { upstreamJobs } from "./upstream.ts";

const changed = (name: "native" | "lint" | "dependencies") =>
  eq(needsOutput<string>("affected", name), "true");
const nativeJobs = ["asan", "ubsan", "fedora_x86", "fedora_arm"] as const;

function nativeJob(
  name: string,
  mode: "ASAN" | "UBSAN" | "fedora",
  runner: string,
): GitHubStepWorkflowJob {
  return job({
    name,
    needs: ["affected", "workflows"],
    if: changed("native"),
    "runs-on": runner,
    "timeout-minutes": 60,
    steps: [checkout, uses(native, { with: { mode } })],
  });
}

const acceptedLane = (id: string, selection: "native" | "lint" | "dependencies") =>
  or(
    and(changed(selection), eq(needsResult(id), "success")),
    and(eq(needsOutput<string>("affected", selection), "false"), eq(needsResult(id), "skipped")),
  );

export const accepted = and(
  eq(needsResult("affected"), "success"),
  eq(needsResult("workflows"), "success"),
  acceptedLane("lint", "lint"),
  or(
    eq(needsResult("dependencies"), "success"),
    and(
      eq(needsResult("dependencies"), "skipped"),
      or(
        eq(needsOutput<string>("affected", "dependencies"), "false"),
        eq(github.eventName, "push"),
        eq(github.eventName, "workflow_dispatch"),
      ),
    ),
  ),
  ...nativeJobs.map((id) => acceptedLane(id, "native")),
);

export const ci = workflow(
  {
    name: "Build and Test",
    on: {
      workflow_dispatch: {},
      pull_request: {},
      merge_group: { types: ["checks_requested"] },
      push: { branches: ["main", "test/*"] },
    },
    permissions: { contents: "read" },
    concurrency: {
      group: format("{0}-{1}", github.workflow, github.ref),
      "cancel-in-progress": expr<boolean>("github.ref != 'refs/heads/main'"),
    },
    jobs: {
      affected: job({
        name: "Affected paths",
        "runs-on": "ubuntu-24.04",
        "timeout-minutes": 5,
        outputs: {
          native: stepOutput<string>("paths", "native"),
          lint: stepOutput<string>("paths", "lint"),
          dependencies: stepOutput<string>("paths", "dependencies"),
        },
        steps: [
          { ...checkout, with: { ...checkout.with, "fetch-depth": 0 } },
          uses(affected, {
            id: "paths",
            with: {
              event: github.eventName,
              base: expr<string>(
                "github.event.pull_request.base.sha || github.event.merge_group.base_sha || github.event.before",
              ),
              head: expr<string>(
                "github.event.pull_request.head.sha || github.event.merge_group.head_sha || github.sha",
              ),
            },
          }),
        ],
      }),
      workflows: job({
        name: "Generated workflows",
        "runs-on": "ubuntu-24.04",
        "timeout-minutes": 10,
        steps: [
          checkout,
          node,
          install,
          {
            name: "Check typed workflows",
            run: command({ file: "npm", args: ["run", "ci", "check"] }),
          },
        ],
      }),
      dependencies: job({
        name: "Dependency review",
        needs: "affected",
        if: and(
          changed("dependencies"),
          or(eq(github.eventName, "pull_request"), eq(github.eventName, "merge_group")),
        ),
        "runs-on": "ubuntu-24.04",
        "timeout-minutes": 10,
        steps: [
          checkout,
          {
            uses: "actions/dependency-review-action@a1d282b36b6f3519aa1f3fc636f609c47dddb294",
            with: {
              "fail-on-severity": "high",
              "base-ref": expr<string>(
                "github.event.pull_request.base.sha || github.event.merge_group.base_sha",
              ),
              "head-ref": expr<string>(
                "github.event.pull_request.head.sha || github.event.merge_group.head_sha",
              ),
            },
          },
        ],
      }),
      lint: job({
        name: "Lint",
        needs: "affected",
        if: changed("lint"),
        "runs-on": "ubuntu-24.04",
        "timeout-minutes": 10,
        steps: [checkout, { uses: "pre-commit/action@v3.0.1" }],
      }),
      asan: nativeJob("Linux Sanitizer Tests (ASAN)", "ASAN", "ubuntu-22.04-arm"),
      ubsan: nativeJob("Linux Sanitizer Tests (UBSAN)", "UBSAN", "ubuntu-22.04-arm"),
      fedora_x86: nativeJob("Linux Fedora (x86_64)", "fedora", "blacksmith-4vcpu-ubuntu-2404"),
      fedora_arm: nativeJob("Linux Fedora (aarch64)", "fedora", "ubuntu-22.04-arm"),
      ...upstreamJobs,
      check_lint: job({
        name: "Check Lint",
        if: always(),
        needs: ["affected", "workflows", "dependencies", "lint", ...nativeJobs],
        "runs-on": "ubuntu-24.04",
        "timeout-minutes": 5,
        permissions: {},
        steps: [
          {
            name: "Accept required checks",
            if: accepted,
            run: command({ file: "true", args: [] }),
          },
          {
            name: "Reject failed or missing checks",
            if: not(accepted),
            run: command({ file: "false", args: [] }),
          },
        ],
      }),
    },
  },
  { filename: "build_and_test.yml" },
);
