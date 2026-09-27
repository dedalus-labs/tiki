import { command, workflow } from "@dedalus-labs/hollywood";

export const update_bypass_list = workflow(
  {
    name: "Update bypass list",
    on: {
      workflow_dispatch: null,
      pull_request_target: {
        types: ["closed"],
      },
      schedule: [
        {
          cron: "33 6 * * *",
        },
      ],
    },
    permissions: {
      contents: "write",
    },
    jobs: {
      update_bypass_list: {
        name: "Update bypass list",
        "runs-on": "ubuntu-22.04",
        steps: [
          {
            uses: "actions/checkout@v7",
          },
          {
            env: {
              GITHUB_TOKEN: "${{ secrets.ZCBENZ_TOKEN_UPDATE_BYPASS_LIST }}",
            },
            run: command({
              file: "node",
              args: [".github/scripts/update-bypass-list.js", "ml-explore", "mlx"],
            }),
          },
        ],
        if: "github.repository == 'ml-explore/mlx'",
      },
    },
  },
  { filename: "update_bypass_list.yml" },
);
