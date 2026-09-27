class ArrayNamespaceInfo:
    def capabilities(self):
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
            "max dimensions": 10,
        }

    def default_device(self):
        import tiki as tk

        return tk.default_device()

    def default_dtypes(self, *, device=None):
        import tiki as tk

        if device is not None and not isinstance(device, tk.Device):
            raise TypeError("Expected a tiki Device")
        return {
            "real floating": tk.float32,
            "complex floating": tk.complex64,
            "integral": tk.int32,
            "indexing": tk.int32,
        }

    def devices(self):
        import tiki as tk

        devices = [
            tk.Device(dev_type, i)
            for dev_type in (tk.cpu, tk.gpu)
            for i in range(tk.device_count(dev_type))
        ]
        return tuple(devices)

    def dtypes(self, *, device=None, kind=None):
        import tiki as tk

        if device is not None and not isinstance(device, tk.Device):
            raise TypeError("Expected a tiki Device")
        device = device if device is not None else self.default_device()

        dtypes = {
            "bool": tk.bool_,
            "int8": tk.int8,
            "int16": tk.int16,
            "int32": tk.int32,
            "int64": tk.int64,
            "uint8": tk.uint8,
            "uint16": tk.uint16,
            "uint32": tk.uint32,
            "uint64": tk.uint64,
            "float32": tk.float32,
            "complex64": tk.complex64,
        }
        if device.type == tk.cpu:
            dtypes["float64"] = tk.float64
        if kind is None:
            return dtypes

        signed = {"int8", "int16", "int32", "int64"}
        unsigned = {"uint8", "uint16", "uint32", "uint64"}
        real = {"float32", "float64"}
        complex_ = {"complex64"}
        kinds = {
            "bool": {"bool"},
            "signed integer": signed,
            "unsigned integer": unsigned,
            "integral": signed | unsigned,
            "real floating": real,
            "complex floating": complex_,
            "numeric": signed | unsigned | real | complex_,
        }
        kind = (kind,) if isinstance(kind, str) else kind
        if not isinstance(kind, tuple) or any(k not in kinds for k in kind):
            raise ValueError(f"Unsupported dtype kind: {kind!r}")
        names = {name for k in kind for name in kinds[k]}
        return {name: dtype for name, dtype in dtypes.items() if name in names}


def __array_namespace_info__():
    return ArrayNamespaceInfo()
