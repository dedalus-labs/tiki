# Copyright © 2026 Dedalus Labs, Inc.

"""Engines, the storage half of a Tiki tensor.

An Engine maps an element offset to a value, and ``engine + delta`` is the
Engine displaced by ``delta`` elements. ``ArrayEngine`` is a flat Tiki buffer
plus an element offset. Element reads and writes go through Tiki indexing,
which is exact but slow. A whole layout is realized as one zero-copy view by
``tiki.layout.tensor.realize``.

``Ptr`` and ``Array`` address foreign and owned ctypes memory,
``ImplicitAccessor`` returns its offsets as values, and ``TransformAccessor``
maps the values another Engine reads. Only ``ArrayEngine`` checks bounds.
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Callable
from operator import index
from typing import Any, SupportsIndex, TypeAlias

import tiki as tk
from tiki.layout._cute import LayoutError
from tiki.layout.tuples import Offset

Scalar: TypeAlias = bool | int | float | complex
CType: TypeAlias = "type[ctypes._SimpleCData[Any]]"


class Engine(ABC):
    """Read-only storage: ``engine[offset]`` reads, ``engine + delta`` displaces."""

    @abstractmethod
    def __add__(self, delta: Any) -> Engine: ...

    @abstractmethod
    def __getitem__(self, offset: Any) -> Any: ...


class MutableEngine(Engine):
    """Storage that ``engine[offset] = value`` also writes."""

    @abstractmethod
    def __setitem__(self, offset: Any, value: Any) -> None: ...


# ``struct`` format characters of the buffer protocol and their ctypes element types.
_FORMATS: dict[str, CType] = {
    "?": ctypes.c_bool,
    "c": ctypes.c_char,
    "b": ctypes.c_byte,
    "B": ctypes.c_ubyte,
    "h": ctypes.c_short,
    "H": ctypes.c_ushort,
    "i": ctypes.c_int,
    "I": ctypes.c_uint,
    "l": ctypes.c_long,
    "L": ctypes.c_ulong,
    "q": ctypes.c_longlong,
    "Q": ctypes.c_ulonglong,
    "n": ctypes.c_ssize_t,
    "N": ctypes.c_size_t,
    "f": ctypes.c_float,
    "d": ctypes.c_double,
    "P": ctypes.c_void_p,
}


def _address(source: Any) -> int:
    """Address of the first element of a ctypes object, an ``array.array``, a NumPy array, or a writable buffer."""
    try:
        return ctypes.addressof(source)
    except TypeError:
        pass
    if hasattr(source, "buffer_info"):
        return int(source.buffer_info()[0])
    if hasattr(source, "__array_interface__"):
        return int(source.__array_interface__["data"][0])
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(source))
    except TypeError as error:
        raise LayoutError(
            f"Ptr cannot take the address of a {type(source).__name__}. Pass an address and a dtype"
        ) from error


def _element(source: Any) -> CType | None:
    if isinstance(source, ctypes.Array):
        return source._type_
    try:
        return _FORMATS.get(memoryview(source).format)
    except TypeError:
        return None


class Ptr(MutableEngine):
    """A typed pointer into memory that another object owns.

    ``source`` is an integer address with an explicit ``dtype``, or an object
    that exposes its storage, such as an ``array.array`` or a ctypes array,
    whose address and element type are inferred. Element ``i`` lives at
    ``address + i * sizeof(dtype)``. Nothing is allocated, copied, or
    bounds-checked, so a ``Ptr`` is exactly as valid as its address. The
    source object is retained as ``owner`` so that its storage stays alive.
    """

    def __init__(
        self, source: Any, dtype: CType | None = None, owner: object = None
    ) -> None:
        if isinstance(source, int):
            if dtype is None:
                raise LayoutError(f"Ptr({source:#x}) needs a dtype for a raw address")
            self.address = source
            self.owner = owner
        else:
            self.address = _address(source)
            self.owner = source if owner is None else owner
            dtype = _element(source) if dtype is None else dtype
            if dtype is None:
                raise LayoutError(
                    f"Ptr cannot infer the element type of a {type(source).__name__}. Pass dtype"
                )
        self.dtype: CType = dtype
        self._pointer = ctypes.cast(self.address, ctypes.POINTER(dtype))

    @property
    def base(self) -> object:
        """The object that owns the storage, or this pointer when nothing does."""
        return self if self.owner is None else self.owner

    def __add__(self, delta: SupportsIndex) -> Ptr:
        address = self.address + index(delta) * ctypes.sizeof(self.dtype)
        return Ptr(address, self.dtype, owner=self.base)

    def __getitem__(self, offset: SupportsIndex) -> Any:
        return self._pointer[index(offset)]

    def __setitem__(self, offset: SupportsIndex, value: Any) -> None:
        self._pointer[index(offset)] = self.dtype(value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ptr):
            return NotImplemented
        return self.address == other.address and self.dtype == other.dtype

    def __repr__(self) -> str:
        return f"Ptr({self.address:#018x}, {self.dtype.__name__})"


class Array(Ptr):
    """A ``Ptr`` to ``size`` elements of ``dtype`` that it allocates and owns."""

    def __init__(self, size: int, dtype: CType = ctypes.c_double) -> None:
        self._storage = (dtype * size)()
        super().__init__(ctypes.addressof(self._storage), dtype)

    def __repr__(self) -> str:
        return f"Array({self.address:#018x}, {self.dtype.__name__})"


class ImplicitAccessor(Engine):
    """An Engine without storage: offset ``i`` reads as ``base + i``.

    Over a layout with coordinate strides, it reads back the coordinate that
    reaches each position, as ``identity_tensor`` does.
    """

    def __init__(self, base: Offset) -> None:
        self.base = base

    def __add__(self, delta: Offset) -> ImplicitAccessor:
        return ImplicitAccessor(self.base + delta)  # type: ignore[operator]

    def __getitem__(self, offset: Offset) -> Offset:
        return self.base + offset  # type: ignore[operator]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImplicitAccessor):
            return NotImplemented
        return bool(self.base == other.base)

    def __repr__(self) -> str:
        return f"{{{self.base}}}"


class TransformAccessor(Engine):
    """Reads ``transform(engine[offset])``. It is read-only, since a transform has no inverse here."""

    def __init__(self, accessor: Engine, transform: Callable[[Any], Any]) -> None:
        self.accessor = accessor
        self.transform = transform

    def __add__(self, delta: Any) -> TransformAccessor:
        return TransformAccessor(self.accessor + delta, self.transform)

    def __getitem__(self, offset: Any) -> Any:
        return self.transform(self.accessor[offset])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransformAccessor):
            return NotImplemented
        return self.accessor == other.accessor and self.transform == other.transform

    def __repr__(self) -> str:
        return f"TransformAccessor({self.accessor}, {self.transform})"


def integer_offset(value: SupportsIndex) -> int:
    """Require an integral Engine displacement without coercing another algebra."""
    try:
        if isinstance(value, bool):
            raise TypeError("boolean offset")
        return index(value)
    except TypeError as error:
        raise LayoutError(f"Engine offsets must be integers, got {value!r}") from error


class ArrayEngine(MutableEngine):
    """A flat Tiki array and an element offset into it.

    ``from_array`` normalizes a base explicitly so its layout is authoritative
    when constructing an ``as_strided`` view.
    """

    def __init__(self, base: tk.array, offset: int = 0) -> None:
        if not isinstance(base, tk.array):
            raise LayoutError("ArrayEngine needs an Tiki array")
        if base.ndim != 1:
            raise LayoutError(
                f"ArrayEngine needs a flat base array, got shape {base.shape}"
            )
        self.base = base
        self.offset = integer_offset(offset)
        if not 0 <= self.offset <= base.size:
            raise LayoutError(
                f"Engine offset {self.offset} is outside [0, {base.size}]"
            )

    def __add__(self, delta: SupportsIndex) -> ArrayEngine:
        return ArrayEngine(self.base, self.offset + integer_offset(delta))

    def _position(self, value: SupportsIndex) -> int:
        position = self.offset + integer_offset(value)
        if not 0 <= position < self.base.size:
            raise LayoutError(
                f"Engine address {position} is outside [0, {self.base.size})"
            )
        return position

    def __getitem__(self, index: SupportsIndex) -> Scalar:
        return self.base[self._position(index)].item()

    def __setitem__(self, index: SupportsIndex, value: Scalar) -> None:
        self.base[self._position(index)] = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayEngine):
            return NotImplemented
        return self.base is other.base and self.offset == other.offset

    def __repr__(self) -> str:
        return f"ArrayEngine({self.base.dtype}[{self.base.size}] + {self.offset})"
