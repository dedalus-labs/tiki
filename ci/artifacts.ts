import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { action, pathInput } from "@dedalus-labs/hollywood/action-runtime";

export const listWheels = action({
  name: "Inspect wheels",
  description: "List the contents of each built wheel",
  localActionPath: "ci/list-wheels",
  inputs: { directory: pathInput({ description: "Wheel directory", default: "wheelhouse" }) },
  outputs: {},
  run: async ({ exec, input }) => {
    const wheels = (await readdir(input.directory)).filter((file) => file.endsWith(".whl")).sort();
    if (wheels.length === 0) throw new Error(`No wheels found in ${input.directory}`);
    for (const wheel of wheels) await exec("unzip", ["-l", join(input.directory, wheel)]);
    return {};
  },
});
