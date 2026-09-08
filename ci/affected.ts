import {
  action,
  choiceInput,
  stringInput,
  stringOutput,
  type ActionInputValues,
  type ScriptExec,
} from "@dedalus-labs/hollywood/action-runtime";

const inputs = {
  event: choiceInput({
    description: "Workflow event",
    options: ["pull_request", "merge_group", "push", "workflow_dispatch"] as const,
  }),
  base: stringInput({ description: "Event base commit", default: "" }),
  head: stringInput({ description: "Event head commit", default: "" }),
} as const;

export type ChangeInput = ActionInputValues<typeof inputs>;
export type Affected = Readonly<{ native: boolean; lint: boolean; dependencies: boolean }>;

const nativeNeutral = [
  /^(docs|examples|benchmarks|\.vscode)\//,
  /^[^/]+\.md$/,
  /^(LICENSE|ACKNOWLEDGMENTS\.md|CITATION\.cff|\.gitignore|\.npmrc)$/,
  /^\.github\/(rulesets\/|ISSUE_TEMPLATE\/|CODEOWNERS$|dependabot\.yml$|pull_request_template\.md$|commit_template\.txt$|workflows\/)/,
  /^ci\/(?!native\.ts$)/,
  /^(package(?:-lock)?\.json|tsconfig\.json)$/,
];

export function classify(files: readonly string[]): Affected {
  return {
    native: files.some((file) => !nativeNeutral.some((pattern) => pattern.test(file))),
    lint: files.some((file) =>
      /\.(py|cpp|h|cmake)$|(^|\/)CMakeLists\.txt$|^\.pre-commit-config\.yaml$/.test(file),
    ),
    dependencies: files.some((file) =>
      /(^|\/)(package(?:-lock)?\.json|Cargo\.(toml|lock)|pyproject\.toml|setup\.py|requirements[^/]*\.txt|[^/]*lock\.ya?ml)$/.test(
        file,
      ),
    ),
  };
}

function commit(value: string): string {
  if (!/^[0-9a-f]{40}$/.test(value) || /^0+$/.test(value)) {
    throw new Error(`Expected a nonzero full commit SHA, received ${JSON.stringify(value)}`);
  }
  return value;
}

export async function affectedForEvent(exec: ScriptExec, input: ChangeInput): Promise<Affected> {
  if (
    input.event === "workflow_dispatch" ||
    (input.event === "push" && /^0{40}$/.test(input.base))
  ) {
    return { native: true, lint: true, dependencies: true };
  }
  const head = commit(input.head);
  const base =
    input.event === "pull_request"
      ? commit(
          (
            await exec("git", ["merge-base", commit(input.base), head], { output: "capture" })
          ).stdout.trim(),
        )
      : commit(input.base);
  const result = await exec(
    "git",
    ["diff", "--name-only", "--no-renames", "-z", base, head, "--"],
    { output: "capture" },
  );
  return classify(result.stdout.split("\0").filter((file) => file.length > 0));
}

export const affected = action({
  name: "Affected paths",
  description: "Select CI from a complete Git diff",
  localActionPath: "ci/affected",
  inputs,
  outputs: {
    native: stringOutput({ description: "Run native builds and tests" }),
    lint: stringOutput({ description: "Run source formatters" }),
    dependencies: stringOutput({ description: "Review dependency changes" }),
  },
  run: async ({ exec, input, log }) => {
    const result = await affectedForEvent(exec, input);
    log.info(JSON.stringify(result));
    return {
      native: String(result.native),
      lint: String(result.lint),
      dependencies: String(result.dependencies),
    };
  },
});
