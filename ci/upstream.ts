import { command, type GitHubWorkflow } from "@dedalus-labs/hollywood";

// These inherited jobs require MLX upstream runners and remain upstream-only.
export const upstreamJobs = {
  build_and_test: {
    name: "${{ matrix.os }} (${{ matrix.toolkit }}, ${{ matrix.arch }})",
    if: "github.repository == 'ml-explore/mlx'",
    needs: "lint",
    strategy: {
      "fail-fast": false,
      matrix: {
        os: ["Linux", "Windows"],
        arch: ["x86_64", "aarch64"],
        toolkit: ["cpu", "cuda-12.6", "cuda-12.9", "cuda-13.0"],
        exclude: [
          {
            os: "Windows",
            arch: "aarch64",
            toolkit: "cuda-12.6",
          },
          {
            os: "Windows",
            arch: "aarch64",
            toolkit: "cuda-12.9",
          },
          {
            os: "Windows",
            arch: "aarch64",
            toolkit: "cuda-13.0",
          },
          {
            os: "Windows",
            arch: "x86_64",
            toolkit: "cuda-12.6",
          },
        ],
      },
    },
    "runs-on":
      "${{ case(matrix.os == 'Windows',\n         case(matrix.arch == 'aarch64', 'windows-11-arm',\n              startsWith(matrix.toolkit, 'cuda'), 'windows-2022-large',\n              'windows-2022'),\n         case(matrix.arch == 'x86_64' && startsWith(matrix.toolkit, 'cuda'), 'gpu-t4-4-core',\n              matrix.arch == 'aarch64', 'ubuntu-22.04-arm',\n              'ubuntu-22.04'))\n}}",
    steps: [
      {
        uses: "actions/checkout@v7",
      },
      {
        uses: "./.github/actions/setup",
        id: "setup",
        with: {
          toolkit: "${{ matrix.toolkit }}",
        },
      },
      {
        uses: "./.github/actions/build",
        with: {
          "cmake-args": "${{ steps.setup.outputs.cmake-args }}",
          debug: "${{ matrix.os != 'Windows' }}",
        },
      },
      {
        name: "Check generated Python stubs with ty",
        if: "matrix.os == 'Linux' && matrix.arch == 'x86_64' && matrix.toolkit == 'cpu'",
        run: command({ file: "uvx", args: ["ty", "check", "python/mlx/core"] }),
      },
      {
        uses: "./.github/actions/test-linux",
        if: "matrix.os == 'Linux' && (matrix.toolkit == 'cpu' || matrix.arch == 'x86_64')",
      },
      {
        uses: "./.github/actions/test-windows",
        if: "matrix.os == 'Windows' && matrix.toolkit == 'cpu'",
      },
    ],
  },
  mac_build: {
    name: "macOS (${{ matrix.macos-target }}, ${{ matrix.toolkit }})",
    if: "github.repository == 'ml-explore/mlx'",
    strategy: {
      "fail-fast": false,
      matrix: {
        "macos-target": ["14.0", "15.0", "26.2"],
        toolkit: ["cpu", "metal", "jit"],
      },
    },
    "runs-on": "macos-26-xlarge",
    needs: "lint",
    steps: [
      {
        uses: "actions/checkout@v7",
      },
      {
        uses: "./.github/actions/setup",
        id: "setup",
        with: {
          toolkit: "${{ matrix.toolkit }}",
          "ccache-key": "test-${{ matrix.macos-target }}",
          "ccache-save": "${{ matrix.toolkit == 'metal' }}",
          "ccache-toolkit": "metal",
        },
      },
      {
        uses: "./.github/actions/build-macos",
        with: {
          "cmake-args": "${{ steps.setup.outputs.cmake-args }}",
          "macos-target": "${{ matrix.macos-target }}",
        },
      },
      {
        uses: "./.github/actions/test-macos",
        if: "matrix.toolkit == 'cpu'",
        with: {
          toolkit: "cpu",
        },
      },
      {
        uses: "actions/upload-artifact@v7",
        if: "matrix.toolkit != 'cpu'",
        with: {
          name: "mlx-${{ matrix.toolkit }}-macos${{ matrix.macos-target }}",
          path: "dist/mlx-*.whl\nbuild/mlx/backend/metal/kernels/mlx.metallib\nbuild/tests/tests\n",
          "if-no-files-found": "error",
        },
      },
    ],
  },
  mac_test: {
    name: "Test macOS (${{ matrix.toolkit }})",
    if: "github.repository == 'ml-explore/mlx'",
    "runs-on": ["self-hosted", "macos"],
    needs: "mac_build",
    strategy: {
      "fail-fast": false,
      matrix: {
        toolkit: ["metal", "jit"],
      },
    },
    steps: [
      {
        uses: "actions/checkout@v7",
      },
      {
        uses: "./.github/actions/setup",
        with: {
          toolkit: "${{ matrix.toolkit }}",
          "use-ccache": false,
        },
      },
      {
        uses: "actions/download-artifact@v8",
        with: {
          path: "artifact",
          pattern: "mlx-${{ matrix.toolkit }}-*",
        },
      },
      {
        run: command({ file: "ls", args: ["-lhR", "artifact"] }),
      },
      {
        uses: "./.github/actions/test-macos",
        with: {
          toolkit: "${{ matrix.toolkit }}",
        },
      },
    ],
  },
  build_documentation: {
    name: "Build Documentation",
    if: "github.repository == 'ml-explore/mlx'",
    "runs-on": "ubuntu-22.04",
    needs: "lint",
    steps: [
      {
        uses: "actions/checkout@v7",
      },
      {
        uses: "./.github/actions/build-docs",
      },
    ],
  },
} satisfies GitHubWorkflow["jobs"];
