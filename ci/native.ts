import { availableParallelism } from "node:os";
import { resolve } from "node:path";

import { action, choiceInput, type ScriptExec } from "@dedalus-labs/hollywood/action-runtime";

export const native = action({
  name: "Native tests",
  description: "Build and test the native CPU backend",
  localActionPath: "ci/native",
  inputs: {
    mode: choiceInput({
      description: "Native test environment",
      options: ["ASAN", "UBSAN", "fedora"] as const,
    }),
  },
  outputs: {},
  run: async ({ exec, input }) => {
    if (input.mode === "fedora") {
      await fedora(exec, process.cwd());
    } else {
      await sanitizer(exec, input.mode);
    }
    return {};
  },
});

export async function sanitizer(exec: ScriptExec, mode: "ASAN" | "UBSAN"): Promise<void> {
  await exec("sudo", ["apt-get", "update", "-y"]);
  await exec(
    "sudo",
    [
      "apt-get",
      "install",
      "-y",
      "build-essential",
      "libblas-dev",
      "liblapacke-dev",
      "libopenblas-dev",
      "cmake",
      "clang",
      "git",
    ],
    {
      env: { DEBIAN_FRONTEND: "noninteractive" },
    },
  );
  const build = `build/ci-${mode.toLowerCase()}`;
  await exec("cmake", [
    "-S",
    ".",
    "-B",
    build,
    "-DCMAKE_BUILD_TYPE=Debug",
    "-DMLX_BUILD_METAL=OFF",
    "-DCMAKE_COMPILE_WARNING_AS_ERROR=ON",
    `-DUSE_${mode}=ON`,
  ]);
  await exec("cmake", ["--build", build, "--parallel", String(availableParallelism())]);
  await exec(`${build}/tests/tests`, [], {
    env:
      mode === "ASAN"
        ? { ASAN_OPTIONS: "detect_leaks=0" }
        : { UBSAN_OPTIONS: "halt_on_error=0:print_stacktrace=1" },
  });
}

export async function fedora(exec: ScriptExec, workspace: string): Promise<void> {
  const created = await exec(
    "docker",
    [
      "create",
      "--volume",
      `${resolve(workspace)}:/workspace`,
      "--workdir",
      "/workspace",
      "fedora:42",
      "sleep",
      "infinity",
    ],
    { output: "capture" },
  );
  const container = created.stdout.trim();
  if (!/^[0-9a-f]{64}$/.test(container)) throw new Error("Docker did not return a container ID");
  try {
    await exec("docker", ["start", container]);
    const inside: ScriptExec = (file, args, options) =>
      exec("docker", ["exec", container, file, ...args], options);
    await inside("dnf", ["update", "-y"]);
    await inside("dnf", [
      "install",
      "-y",
      "blas-devel",
      "lapack-devel",
      "openblas-devel",
      "make",
      "cmake",
      "clang",
      "git",
    ]);
    await inside("cmake", [
      "-S",
      ".",
      "-B",
      "build/ci-fedora",
      "-DMLX_BUILD_METAL=OFF",
      "-DCMAKE_BUILD_TYPE=Debug",
    ]);
    await inside("cmake", [
      "--build",
      "build/ci-fedora",
      "--parallel",
      String(availableParallelism()),
    ]);
    await inside("./build/ci-fedora/tests/tests", []);
  } finally {
    await exec("docker", ["rm", "--force", container]);
  }
}
