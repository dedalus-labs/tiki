# Copyright © 2026 Dedalus Labs, Inc.

"""Derivative checks for Tiki ops, in the shape of ``jax.test_util.check_grads``.

``check_grads`` tests the transforms registered on ``f`` against evidence that
does not depend on them:

- ``fwd``: the JVP along a random direction against a float64 central
  difference on the CPU stream, or against ``reference``'s JVP.
- ``rev``: the VJP against the JVP through the transpose identity
  ``u . (J v) == (J^T u) . v``, which is exact and needs no step size, and
  against ``reference``'s VJP when one is given.
- ``vmap``: the batched forward and the batched VJP against per-example runs.

The VJP is also evaluated twice and must agree within ``nondeterminism``.
Every check runs again on transposed, offset, and reversed views of the
arguments, because Tiki specializes kernels on strides. Higher orders recurse
on the JVP and VJP with fixed directions, so ``order=2`` checks that the
derivative programs are themselves differentiable. A random direction shows
that a derivative is wrong in a few evaluations at any size; ``full_jacobian``
then shows which element.

Tolerances assume the dtype's own arithmetic. A CUDA float32 matmul may run
in reduced precision (TF32), so check matmul-bearing functions in float64 on
the CPU stream or through ``reference``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

Tree = Any
Mode = Literal["fwd", "rev", "vmap"]
Leaves = list[mx.array]
MODES: frozenset[str] = frozenset({"fwd", "rev", "vmap"})
LAYOUTS = ("transposed", "offset", "reversed")
BATCH = 2


class GradcheckError(AssertionError):
    """A registered transform disagrees with evidence independent of it."""


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


TOLERANCES = {
    mx.float16: Tolerance(1e-2, 1e-2),
    mx.bfloat16: Tolerance(5e-2, 5e-2),
    mx.float32: Tolerance(1e-3, 1e-3),
    mx.float64: Tolerance(1e-5, 1e-3),
}


class Flat:
    """``function`` over flat leaves; remembers the input and output structures."""

    def __init__(self, function: Callable[..., Tree], args: tuple[Tree, ...]):
        self.function = function
        self.paths = [path for path, _ in tree_flatten(list(args))]
        self.outputs: list[str] = []

    def __call__(self, *leaves: mx.array) -> Leaves:
        args = tree_unflatten(list(zip(self.paths, leaves)))
        flat = tree_flatten(self.function(*args))
        self.outputs = [path for path, _ in flat]
        return [leaf for _, leaf in flat]


def leaves_of(args: tuple[Tree, ...]) -> Leaves:
    return [leaf for _, leaf in tree_flatten(list(args))]


def derive(flat: Flat | None, function: Callable[..., Leaves]) -> Flat | None:
    """A function of the same structured arguments as ``flat``."""
    if flat is None:
        return None
    derived = Flat(function, ())
    derived.paths = flat.paths
    return derived


def fixed_jvp(flat: Flat | None, tangents: Leaves) -> Flat | None:
    """The JVP with fixed tangents as a function of the primals."""
    return derive(flat, lambda *args: mx.jvp(flat, leaves_of(args), tangents)[1])


def fixed_vjp(flat: Flat | None, cotangents: Leaves) -> Flat | None:
    """The VJP with fixed cotangents as a function of the primals."""
    return derive(flat, lambda *args: mx.vjp(flat, leaves_of(args), cotangents)[1])


def to_float64(leaves: Leaves) -> Leaves:
    with mx.stream(mx.cpu):
        return [leaf.astype(mx.float64) for leaf in leaves]


def inner(left: Leaves, right: Leaves) -> tuple[float, float]:
    """Sum of termwise products and of their magnitudes, accumulated in float64."""
    total, magnitude = 0.0, 0.0
    with mx.stream(mx.cpu):
        for a, b in zip(to_float64(left), to_float64(right)):
            products = a * b
            total += products.sum().item()
            magnitude += mx.abs(products).sum().item()
    return total, magnitude


def view(leaf: mx.array, layout: str) -> mx.array:
    """The same values through a different storage map; rank-0 leaves have one."""
    if leaf.ndim == 0 or (layout == "transposed" and leaf.ndim < 2):
        return leaf
    if layout == "transposed":
        return mx.contiguous(leaf.T).T
    if layout == "offset":
        pad = mx.zeros((*leaf.shape[:-1], 1), dtype=leaf.dtype)
        return mx.concatenate([pad, leaf], axis=-1)[..., 1:]
    return mx.contiguous(leaf[::-1])[::-1]


class Checker:
    def __init__(
        self,
        modes: frozenset[str],
        eps: float,
        atol: float | None,
        rtol: float | None,
        nondeterminism: float,
        full_jacobian: bool,
        key: mx.array,
    ):
        self.modes = modes
        self.eps = eps
        self.atol = atol
        self.rtol = rtol
        self.nondeterminism = nondeterminism
        self.full_jacobian = full_jacobian
        self.key = key

    def tolerance(self, dtype: mx.Dtype) -> Tolerance:
        default = TOLERANCES[dtype]
        return Tolerance(
            default.atol if self.atol is None else self.atol,
            default.rtol if self.rtol is None else self.rtol,
        )

    def directions(self, like: Leaves, batch: int = 0) -> Leaves:
        """Fresh normal directions shaped like ``like``; a batch axis is prepended."""
        self.key, *keys = mx.random.split(self.key, len(like) + 1)
        with mx.stream(mx.cpu):
            return [
                mx.random.normal(
                    (batch, *leaf.shape) if batch else leaf.shape,
                    dtype=leaf.dtype,
                    key=key,
                )
                for leaf, key in zip(like, keys)
            ]

    def compare(
        self,
        actual: Leaves,
        expected: Leaves,
        paths: Sequence[str],
        what: str,
        tolerance: Tolerance | None = None,
    ) -> None:
        for path, got, want in zip(paths, actual, expected):
            if got.shape != want.shape:
                raise GradcheckError(
                    f"{what}: {path}: shape {got.shape}, expected {want.shape}"
                )
            bound = tolerance or self.tolerance(got.dtype)
            with mx.stream(mx.cpu):
                got64, want64 = to_float64([got, want])
                error = mx.abs(got64 - want64)
                excess = error - (bound.atol + bound.rtol * mx.abs(want64))
                if excess.size == 0 or excess.max().item() <= 0:
                    continue
                index = mx.argmax(excess).item()
                raise GradcheckError(
                    f"{what}: {path} at flat index {index}: got "
                    f"{got64.reshape(-1)[index].item():.6g}, expected "
                    f"{want64.reshape(-1)[index].item():.6g} (atol {bound.atol:g}, rtol {bound.rtol:g})"
                )

    def central_difference(self, g: Flat, x: Leaves, tangents: Leaves) -> Leaves:
        with mx.stream(mx.cpu):
            x64, v64 = to_float64(x), to_float64(tangents)
            try:
                plus = g(*(leaf + self.eps * t for leaf, t in zip(x64, v64)))
                minus = g(*(leaf - self.eps * t for leaf, t in zip(x64, v64)))
            except Exception as error:
                raise GradcheckError(
                    "finite differences evaluate f in float64 on the CPU stream, which "
                    "failed; pass reference= for a device kernel"
                ) from error
            return [(p - m) / (2 * self.eps) for p, m in zip(plus, minus)]

    def forward(
        self, g: Flat, r: Flat | None, x: Leaves, tangents: Leaves, context: str
    ) -> None:
        actual = mx.jvp(g, x, tangents)[1]
        if r is None:
            expected = self.central_difference(g, x, tangents)
        else:
            expected = mx.jvp(r, x, tangents)[1]
        self.compare(actual, expected, g.outputs, f"{context}: fwd")

    def reverse(
        self,
        g: Flat,
        r: Flat | None,
        x: Leaves,
        tangents: Leaves,
        cotangents: Leaves,
        context: str,
    ) -> None:
        jvp = mx.jvp(g, x, tangents)[1]
        vjp = mx.vjp(g, x, cotangents)[1]
        lhs, lhs_magnitude = inner(cotangents, jvp)
        rhs, rhs_magnitude = inner(vjp, tangents)
        bound = self.tolerance(x[0].dtype)
        allowed = bound.atol + bound.rtol * (lhs_magnitude + rhs_magnitude) / 2
        if abs(lhs - rhs) > allowed:
            raise GradcheckError(
                f"{context}: rev: transpose identity broken: u.(Jv) = {lhs:.6g} but "
                f"(J^T u).v = {rhs:.6g} (allowed {allowed:.3g})"
            )
        if r is not None:
            self.compare(vjp, mx.vjp(r, x, cotangents)[1], g.paths, f"{context}: rev")

    def batched(self, g: Flat, x: Leaves, context: str) -> None:
        stacked = self.directions(x, BATCH)
        actual = mx.vmap(g)(*stacked)
        expected = [
            mx.stack([g(*(leaf[b] for leaf in stacked))[j] for b in range(BATCH)])
            for j in range(len(g.outputs))
        ]
        self.compare(actual, expected, g.outputs, f"{context}: vmap forward")
        cotangents = self.directions(g(*x), BATCH)

        def vjp_of(*us: mx.array) -> Leaves:
            return mx.vjp(g, x, list(us))[1]

        actual = mx.vmap(vjp_of)(*cotangents)
        expected = [
            mx.stack([vjp_of(*(u[b] for u in cotangents))[j] for b in range(BATCH)])
            for j in range(len(x))
        ]
        self.compare(actual, expected, g.paths, f"{context}: vmap vjp")

    def repeatable(self, g: Flat, x: Leaves, cotangents: Leaves, context: str) -> None:
        first = mx.vjp(g, x, cotangents)[1]
        mx.eval(first)
        second = mx.vjp(g, x, cotangents)[1]
        mx.eval(second)
        self.compare(
            second,
            first,
            g.paths,
            f"{context}: nondeterministic vjp",
            Tolerance(self.nondeterminism, 0.0),
        )

    def jacobian(self, g: Flat, r: Flat | None, x: Leaves, context: str) -> None:
        """Every column from the JVP against evidence, and every row from the VJP against the columns."""
        outputs = g(*x)
        columns, evidence = [], []
        for j, leaf in enumerate(x):
            for i in range(leaf.size):
                basis = [mx.zeros_like(other) for other in x]
                basis[j] = mx.zeros(leaf.size, dtype=leaf.dtype)
                basis[j][i] = 1
                basis[j] = basis[j].reshape(leaf.shape)
                columns.append(mx.jvp(g, x, basis)[1])
                if r is None:
                    evidence.append(self.central_difference(g, x, basis))
                else:
                    evidence.append(mx.jvp(r, x, basis)[1])
        rows = []
        for j, leaf in enumerate(outputs):
            for i in range(leaf.size):
                basis = [mx.zeros_like(other) for other in outputs]
                basis[j] = mx.zeros(leaf.size, dtype=leaf.dtype)
                basis[j][i] = 1
                basis[j] = basis[j].reshape(leaf.shape)
                rows.append(mx.vjp(g, x, basis)[1])
        inputs = [
            f"{path}[{i}]" for path, leaf in zip(g.paths, x) for i in range(leaf.size)
        ]
        output_names = [
            f"{path}[{i}]"
            for path, leaf in zip(g.outputs, outputs)
            for i in range(leaf.size)
        ]
        with mx.stream(mx.cpu):
            from_jvp = mx.stack(
                [mx.concatenate([c.reshape(-1) for c in column]) for column in columns],
                axis=1,
            )
            expected = mx.stack(
                [
                    mx.concatenate([c.reshape(-1) for c in column])
                    for column in evidence
                ],
                axis=1,
            )
            from_vjp = mx.stack(
                [mx.concatenate([c.reshape(-1) for c in row]) for row in rows], axis=0
            )
        for name, matrix in (("jvp column", from_jvp), ("vjp row", from_vjp)):
            bound = self.tolerance(matrix.dtype)
            with mx.stream(mx.cpu):
                excess = mx.abs(
                    matrix.astype(mx.float64) - expected.astype(mx.float64)
                ) - (bound.atol + bound.rtol * mx.abs(expected.astype(mx.float64)))
                if excess.size == 0 or excess.max().item() <= 0:
                    continue
                flat = mx.argmax(excess).item()
                o, i = divmod(flat, len(inputs))
                raise GradcheckError(
                    f"{context}: full jacobian: {name} d{output_names[o]}/d{inputs[i]} = "
                    f"{matrix[o, i].item():.6g}, expected {expected[o, i].item():.6g}"
                )

    def run(self, g: Flat, r: Flat | None, x: Leaves, order: int, context: str) -> None:
        tangents = self.directions(x)
        cotangents = self.directions(g(*x))
        if "fwd" in self.modes:
            self.forward(g, r, x, tangents, context)
        if "rev" in self.modes:
            self.reverse(g, r, x, tangents, cotangents, context)
            self.repeatable(g, x, cotangents, context)
        if "vmap" in self.modes:
            self.batched(g, x, context)
        if self.full_jacobian:
            self.jacobian(g, r, x, context)
        if order > 1:
            if "fwd" in self.modes:
                self.run(
                    fixed_jvp(g, tangents),
                    fixed_jvp(r, tangents),
                    x,
                    order - 1,
                    f"{context} > jvp",
                )
            if "rev" in self.modes:
                self.run(
                    fixed_vjp(g, cotangents),
                    fixed_vjp(r, cotangents),
                    x,
                    order - 1,
                    f"{context} > vjp",
                )


def check_grads(
    f: Callable[..., Tree],
    args: tuple[Tree, ...],
    *,
    order: int = 1,
    modes: Sequence[Mode] = ("fwd", "rev", "vmap"),
    reference: Callable[..., Tree] | None = None,
    layouts: bool = True,
    full_jacobian: bool = False,
    nondeterminism: float = 0.0,
    eps: float = 1e-6,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
) -> None:
    """Check the derivatives of ``f`` at ``args``; raise ``GradcheckError`` on the first failure.

    ``args`` is the tuple of positional arguments, each a pytree of floating
    point arrays. ``reference`` is a second implementation of ``f`` with trusted
    derivatives; without it, first-order evidence comes from float64 central
    differences of ``f`` on the CPU stream, which a device-only kernel cannot
    provide.
    """
    if not isinstance(args, tuple):
        raise TypeError("args must be the tuple of positional arguments")
    if order < 1:
        raise ValueError("order must be at least 1")
    unknown = set(modes) - MODES
    if unknown:
        raise ValueError(
            f"unknown modes {sorted(unknown)}; choose from {sorted(MODES)}"
        )
    x = leaves_of(args)
    if not x:
        raise ValueError("args must contain at least one array")
    if any(
        not isinstance(leaf, mx.array) or leaf.dtype not in TOLERANCES for leaf in x
    ):
        raise TypeError("check_grads needs floating point array leaves")
    g = Flat(f, args)
    r = Flat(reference, args) if reference is not None else None
    checker = Checker(
        frozenset(modes),
        eps,
        atol,
        rtol,
        nondeterminism,
        full_jacobian,
        mx.random.key(seed),
    )
    checker.run(g, r, x, order, "dense")
    if not layouts:
        return
    base = g(*x)
    for layout in LAYOUTS:
        leaves = [view(leaf, layout) for leaf in x]
        if all(a is b for a, b in zip(leaves, x)):
            continue
        checker.compare(g(*leaves), base, g.outputs, f"{layout}: forward")
        checker.run(g, r, leaves, order, layout)
