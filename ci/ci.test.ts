import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { data, Evaluator, Lexer, Parser } from "@actions/expressions";
import { accepted, ci } from "./ci.ts";
import { native, fedora, sanitizer } from "./native.ts";
import type { Command, ScriptExec } from "@dedalus-labs/hollywood";

type Result = "success" | "failure" | "cancelled" | "skipped";
const nativeIds = ["asan", "ubsan", "fedora_x86", "fedora_arm"] as const;

function evaluate(
  selected: boolean,
  overrides: Readonly<Record<string, Result>> = {},
  outputs = String(selected),
): boolean {
  const needs = new data.Dictionary();
  for (const id of ["affected", "workflows", "lint", "dependencies", ...nativeIds]) {
    const defaultResult =
      id === "affected" || id === "workflows" || selected ? "success" : "skipped";
    needs.add(
      id,
      new data.Dictionary(
        { key: "result", value: new data.StringData(overrides[id] ?? defaultResult) },
        {
          key: "outputs",
          value: new data.Dictionary(
            ...["native", "lint", "dependencies"].map((key) => ({
              key,
              value: new data.StringData(outputs),
            })),
          ),
        },
      ),
    );
  }
  const tokens = new Lexer(accepted.slice(3, -2)).lex().tokens;
  const expression = new Parser(tokens, ["needs", "github"], []).parse();
  const context = new data.Dictionary(
    { key: "needs", value: needs },
    {
      key: "github",
      value: new data.Dictionary({ key: "event_name", value: new data.StringData("pull_request") }),
    },
  );
  return new Evaluator(expression, context).evaluate().coerceString() === "true";
}

test("required check accepts only proven success or intentional skips", () => {
  assert.equal(evaluate(true), true);
  assert.equal(evaluate(false), true);
  for (const id of ["affected", "workflows", "lint", "dependencies", ...nativeIds]) {
    for (const result of ["failure", "cancelled", "skipped"] as const) {
      assert.equal(evaluate(true, { [id]: result }), false, `${id}: ${result}`);
    }
  }
  assert.equal(evaluate(false, { affected: "failure" }), false);
  assert.equal(evaluate(false, {}, ""), false);
  assert.equal(evaluate(false, { asan: "failure" }), false);
});

test("required check names exist on every PR and merge-group workflow", async () => {
  const ruleset = JSON.parse(await readFile(".github/rulesets/main.json", "utf8")) as {
    rules: { type: string; parameters?: { required_status_checks?: { context: string }[] } }[];
  };
  const names = Object.values(ci.jobs).map((job) => job.name);
  for (const rule of ruleset.rules)
    for (const check of rule.parameters?.required_status_checks ?? [])
      assert.ok(names.includes(check.context), check.context);
  assert.ok(ci.on.pull_request);
  assert.ok(ci.on.merge_group);
  assert.equal(ci.jobs.check_lint?.if, "${{ always() }}");
  assert.equal(native.localActionPath, "ci/native");
});

test("native actions execute structured commands and retire their container after failure", async () => {
  const commands: Command[] = [];
  const container = "a".repeat(64);
  const exec: ScriptExec = async (file, args, options) => {
    commands.push({ file, args, ...options });
    if (args.includes("update")) throw new Error("package installation failed");
    return { exitCode: 0, stdout: container, stderr: "" };
  };
  await assert.rejects(fedora(exec, "/tmp/workspace with spaces"), /package installation failed/);
  assert.deepEqual(commands.at(-1), { file: "docker", args: ["rm", "--force", container] });
  assert.ok(commands[0]?.args.includes("/tmp/workspace with spaces:/workspace"));
  commands.length = 0;
  const success: ScriptExec = async (file, args, options) => {
    commands.push({ file, args, ...options });
    return { exitCode: 0, stdout: "", stderr: "" };
  };
  await sanitizer(success, "ASAN");
  assert.ok(commands.some((entry) => entry.args.includes("-DUSE_ASAN=ON")));
  assert.ok(
    commands.every((entry) => !entry.args.some((arg) => /^-DCMAKE_C(?:XX)?_COMPILER=/.test(arg))),
  );
  assert.ok(commands.every((entry) => !["bash", "sh", "cmd"].includes(entry.file)));
});
