# Instructions for Tiki

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for code, comments, commits, and
communication. AI assistance is welcome under the same standards as human work.

- Understand the relevant implementation before editing it.
- Keep changes small and focused. Preserve unrelated work.
- Test changed behavior and report the commands, results, and remaining limits.
- Use concise comments to explain intent or invariants, not to repeat the code.
- Write descriptions and review replies in plain English. Verify generated text.
- Never claim tests passed or a human wrote content without evidence.
- Push, publish, or reply on a contributor's behalf only when authorized.
- Follow MLX's own contribution policy for submissions to MLX upstream.

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
