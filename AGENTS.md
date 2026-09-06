# Instructions for Tiki

Tiki is an internal derivative of MLX. The `upstream-main` branch mirrors
`ml-explore/mlx`; Tiki changes belong on `main` and feature branches.

## Agent rules

- Never push without explicit user authorization.
- Keep the Tiki patch set small enough to audit against `upstream-main`.
- Send generally useful fixes upstream instead of carrying them indefinitely.
- Follow MLX's contribution policy for changes proposed back to MLX.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Use ASD-STE100 Simplified Technical English, simple wordings

### Examples

```c++
  // Good (no comment)

  std::string module_name =
    fmt::format("{}_{:x}", name_, std::hash<std::string>{}(source_));

  // Bad (excessive comment for explicit code)

  // The module cache is keyed on this name, so it has to include the source:
  // two kernels sharing a name but not a body would otherwise both run
  // whichever was compiled first. Same fix as 3833 on the Metal side.
```
