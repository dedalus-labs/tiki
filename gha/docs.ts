import { command, job, workflow } from "@dedalus-labs/hollywood";
import {
	checkoutAction,
	deployPagesAction,
	setupPythonAction,
	uploadPagesArtifactAction,
} from "./actions";
import { trustedCiRun } from "./guards";

const docsPaths = ["docs/**", "python/**", "experiments/**/*.md", "README.md", "gha/docs.ts"] as const;

export const docs = workflow({
	name: "Docs",
	on: {
		push: { branches: ["main"], paths: docsPaths },
		pull_request: { branches: ["main"], paths: docsPaths },
		workflow_dispatch: {},
	},
	concurrency: {
		group: "pages",
		"cancel-in-progress": false,
	},
	permissions: { contents: "read" },
	jobs: {
		build: job({
			name: "Build",
			if: trustedCiRun,
			"runs-on": "ubuntu-latest",
			steps: [
				{ uses: checkoutAction, with: { "persist-credentials": false } },
				{ uses: setupPythonAction, with: { "python-version": "3.13" } },
				{
					name: "Refresh the package index",
					run: command({ file: "sudo", args: ["apt-get", "update"] }),
				},
				{
					name: "Install Doxygen and BLAS",
					run: command({
						file: "sudo",
						args: [
							"apt-get",
							"install",
							"-y",
							"--no-install-recommends",
							"doxygen",
							"libblas-dev",
							"liblapack-dev",
							"liblapacke-dev",
						],
					}),
				},
				{
					name: "Install Tiki",
					env: { CMAKE_BUILD_PARALLEL_LEVEL: "4" },
					run: command({ file: "python", args: ["-m", "pip", "install", "."] }),
				},
				{
					name: "Install docs dependencies",
					run: command({
						file: "python",
						args: ["-m", "pip", "install", "-r", "docs/requirements.txt"],
					}),
				},
				{
					name: "Index the C++ headers",
					"working-directory": "docs",
					run: command({ file: "doxygen", args: [] }),
				},
				{
					name: "Build the site",
					"working-directory": "docs",
					run: command({
						file: "python",
						args: ["-m", "sphinx", "-W", "--keep-going", "-b", "html", "src", "build/html"],
					}),
				},
				{
					name: "Upload pages artifact",
					uses: uploadPagesArtifactAction,
					with: { path: "docs/build/html" },
				},
			],
		}),
		deploy: job({
			name: "Deploy",
			needs: "build",
			if: "github.event_name == 'push' && github.ref == 'refs/heads/main'",
			"runs-on": "ubuntu-latest",
			permissions: {
				pages: "write",
				"id-token": "write",
			},
			environment: {
				name: "github-pages",
				url: "${{ steps.deployment.outputs.page_url }}",
			},
			steps: [{ id: "deployment", name: "Deploy to GitHub Pages", uses: deployPagesAction }],
		}),
	},
});
