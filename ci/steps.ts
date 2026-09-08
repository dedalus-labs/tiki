import { command } from "@dedalus-labs/hollywood";

export const checkout = {
  uses: "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
  with: { "persist-credentials": false },
} as const;

export const node = {
  uses: "actions/setup-node@48b55a011bda9f5d6aeb4c2d9c7362e8dae4041e",
  with: { "node-version": "24" },
} as const;

export const install = {
  name: "Install workflow dependencies",
  run: command({ file: "npm", args: ["ci", "--ignore-scripts"] }),
} as const;
