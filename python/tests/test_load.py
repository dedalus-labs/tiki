# Copyright © 2023 Apple Inc.

import os
import platform
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tiki as tk
import tiki_tests


class TestLoad(tiki_tests.TIKITestCase):
    dtypes = [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float16",
        "complex64",
    ]

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_and_load(self):
        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_tiki = os.path.join(
                            self.test_dir, f"tiki_{dt}_{i}.npy"
                        )
                        save_file_npy = os.path.join(self.test_dir, f"npy_{dt}_{i}.npy")

                        save_arr = np.random.uniform(0.0, 32.0, size=shape)
                        save_arr_npy = save_arr.astype(getattr(np, dt))
                        save_arr_tiki = tk.array(save_arr_npy)

                        tk.save(save_file_tiki, save_arr_tiki)
                        np.save(save_file_npy, save_arr_npy)

                        # Load array saved by tiki as tiki array
                        load_arr_tiki_tiki = tk.load(save_file_tiki)
                        self.assertTrue(
                            tk.array_equal(load_arr_tiki_tiki, save_arr_tiki)
                        )

                        # Load array saved by numpy as tiki array
                        load_arr_npy_tiki = tk.load(save_file_npy)
                        self.assertTrue(
                            tk.array_equal(load_arr_npy_tiki, save_arr_tiki)
                        )

                        # Load array saved by tiki as numpy array
                        load_arr_tiki_npy = np.load(save_file_tiki)
                        self.assertTrue(np.array_equal(load_arr_tiki_npy, save_arr_npy))

        save_file = os.path.join(self.test_dir, f"tiki_path.npy")
        save_arr = tk.ones((32,))
        tk.save(Path(save_file), save_arr)

        # Load array saved by tiki as tiki array
        load_arr = tk.load(Path(save_file))
        self.assertTrue(tk.array_equal(load_arr, save_arr))

    def test_load_npy_dtype(self):
        save_file = os.path.join(self.test_dir, "tiki_path.npy")
        a = np.random.randn(8).astype(np.float64)
        np.save(save_file, a)
        out = tk.load(save_file, stream=tk.cpu)
        self.assertEqual(out.dtype, tk.float64)
        self.assertTrue(np.array_equal(np.array(out), a))

        a = np.random.randn(8).astype(np.float64)
        b = np.random.randn(8).astype(np.float64)
        c = a + 0j * b
        np.save(save_file, c)
        with self.assertRaises(Exception):
            out = tk.load(save_file, stream=tk.cpu)

    def test_load_npy_read_error(self):
        save_file = os.path.join(self.test_dir, "truncated.npy")
        expected = np.arange(16, dtype=np.float32)
        np.save(save_file, expected)
        with open(save_file, "r+b") as f:
            f.truncate(os.path.getsize(save_file) - expected.nbytes)

        out = tk.load(save_file, stream=tk.cpu)
        with self.assertRaises(RuntimeError):
            tk.eval(out)

    def test_async_load_npy_read_error_across_streams(self):
        save_file = os.path.join(self.test_dir, "truncated_async.npy")
        expected = np.arange(16, dtype=np.float32)
        np.save(save_file, expected)
        with open(save_file, "r+b") as f:
            f.truncate(os.path.getsize(save_file) - expected.nbytes)

        producer_stream = tk.new_stream(tk.cpu)
        consumer_stream = tk.new_stream(tk.cpu)
        out = tk.add(
            tk.load(save_file, stream=producer_stream),
            1.0,
            stream=consumer_stream,
        )
        with self.assertRaises(RuntimeError):
            tk.eval(out)
        # Depending on backend the error might be caught early before poisoning
        # the producer_stream, but still sync to clear the errors.
        try:
            tk.synchronize(producer_stream)
        except Exception:
            pass

    def test_save_over_lazily_loaded_safetensors(self):
        # The save must eval the lazy input before truncating the file it was
        # loaded from. Shift the data so stale content cannot pass.
        save_file = os.path.join(self.test_dir, "resave.safetensors")
        tk.save_safetensors(save_file, {"a": tk.ones((4, 4))})
        a = tk.load(save_file, stream=tk.cpu)["a"]
        tk.save_safetensors(save_file, {"a": a + 1})
        out = tk.load(save_file, stream=tk.cpu)["a"]
        self.assertTrue(np.array_equal(np.array(out), 2 * np.ones((4, 4))))

    def test_save_over_lazily_loaded_npy(self):
        save_file = os.path.join(self.test_dir, "resave.npy")
        tk.save(save_file, tk.ones((4, 4)))
        a = tk.load(save_file, stream=tk.cpu)
        tk.save(save_file, a + 2)
        out = tk.load(save_file, stream=tk.cpu)
        self.assertTrue(np.array_equal(np.array(out), 3 * np.ones((4, 4))))

    def test_save_and_load_safetensors(self):
        test_file = os.path.join(self.test_dir, "test.safetensors")
        with self.assertRaises(Exception):
            tk.save_safetensors(test_file, {"a": tk.ones((4, 4))}, {"testing": 0})

        for obj in [str, Path]:
            tk.save_safetensors(
                obj(test_file),
                {"test": tk.ones((2, 2))},
                {"testing": "test", "format": "tiki"},
            )
            res = tk.load(obj(test_file), return_metadata=True)
            self.assertEqual(len(res), 2)
            self.assertEqual(res[1], {"testing": "test", "format": "tiki"})

        for dt in self.dtypes + ["bfloat16"]:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_tiki = os.path.join(
                            self.test_dir, f"tiki_{dt}_{i}_fs.safetensors"
                        )
                        save_dict = {
                            "test": (
                                tk.random.normal(shape=shape, dtype=getattr(tk, dt))
                                if dt in ["float32", "float16", "bfloat16"]
                                else tk.ones(shape, dtype=getattr(tk, dt))
                            )
                        }

                        with open(save_file_tiki, "wb") as f:
                            tk.save_safetensors(f, save_dict)
                        with open(save_file_tiki, "rb") as f:
                            load_dict = tk.load(f)

                        self.assertTrue("test" in load_dict)
                        self.assertTrue(
                            tk.array_equal(load_dict["test"], save_dict["test"])
                        )

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        # TODO: Add support for other dtypes (self.dtypes + ["bfloat16"])
        supported_dtypes = ["float16", "float32", "int8", "int16", "int32"]
        for dt in supported_dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_tiki = os.path.join(
                            self.test_dir, f"tiki_{dt}_{i}_fs.gguf"
                        )
                        save_dict = {
                            "test": (
                                tk.random.normal(shape=shape, dtype=getattr(tk, dt))
                                if dt in ["float32", "float16", "bfloat16"]
                                else tk.ones(shape, dtype=getattr(tk, dt))
                            )
                        }

                        tk.save_gguf(save_file_tiki, save_dict)
                        load_dict = tk.load(save_file_tiki)

                        self.assertTrue("test" in load_dict)
                        self.assertTrue(
                            tk.array_equal(load_dict["test"], save_dict["test"])
                        )

        save_file_tiki = os.path.join(self.test_dir, f"tiki_path_test_fs.gguf")
        save_dict = {"test": tk.ones(shape)}
        tk.save_gguf(Path(save_file_tiki), save_dict)
        load_dict = tk.load(Path(save_file_tiki))
        self.assertTrue("test" in load_dict)
        self.assertTrue(tk.array_equal(load_dict["test"], save_dict["test"]))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_load_gguf_quantized(self):
        # Write a minimal GGUF v3 file with one quantized tensor and check
        # that the loaded (weight, scales, biases) triplet dequantizes to the
        # values prescribed by the GGML block formats.
        rows, cols = 2, 64
        n_blocks = rows * cols // 32
        np.random.seed(7)

        def write_gguf(path, ggml_type_id, blocks):
            with open(path, "wb") as f:
                f.write(b"GGUF")
                # version, tensor count, metadata kv count
                f.write(struct.pack("<IQQ", 3, 1, 0))
                name = b"tensor.weight"
                f.write(struct.pack("<Q", len(name)) + name)
                # dims are stored in GGML order: ne[0] = cols, ne[1] = rows
                f.write(struct.pack("<IQQIQ", 2, cols, rows, ggml_type_id, 0))
                f.write(b"\x00" * (-f.tell() % 32))  # default alignment
                f.write(blocks.tobytes())

        def packed_nibbles(q):
            # GGML nibble order: byte j = element j | element j + 16 << 4
            return (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)

        d = np.full((n_blocks, 1), 0.5, dtype=np.float16)
        m = np.full((n_blocks, 1), -1.25, dtype=np.float16)
        q4 = np.random.randint(0, 16, size=(n_blocks, 32)).astype(np.uint8)
        q8 = np.random.randint(-128, 128, size=(n_blocks, 32)).astype(np.int8)
        d32 = d.astype(np.float32)
        m32 = m.astype(np.float32)

        # (name, ggml type id, bits, block bytes, expected dequantized values)
        cases = [
            (
                "Q4_0",
                2,
                4,
                np.concatenate([d.view(np.uint8), packed_nibbles(q4)], axis=1),
                d32 * (q4.astype(np.float32) - 8),
            ),
            (
                "Q4_1",
                3,
                4,
                np.concatenate(
                    [d.view(np.uint8), m.view(np.uint8), packed_nibbles(q4)],
                    axis=1,
                ),
                d32 * q4.astype(np.float32) + m32,
            ),
            (
                "Q8_0",
                8,
                8,
                np.concatenate([d.view(np.uint8), q8.view(np.uint8)], axis=1),
                d32 * q8.astype(np.float32),
            ),
        ]
        for name, type_id, bits, blocks, expected in cases:
            with self.subTest(qtype=name):
                save_file = os.path.join(self.test_dir, f"quant_{name}.gguf")
                write_gguf(save_file, type_id, blocks)
                load_dict = tk.load(save_file)
                self.assertEqual(load_dict["tensor.weight"].dtype, tk.uint32)
                self.assertEqual(load_dict["tensor.scales"].shape, (rows, cols // 32))
                self.assertEqual(load_dict["tensor.biases"].shape, (rows, cols // 32))
                dequantized = tk.dequantize(
                    load_dict["tensor.weight"],
                    load_dict["tensor.scales"],
                    load_dict["tensor.biases"],
                    group_size=32,
                    bits=bits,
                )
                self.assertTrue(
                    np.array_equal(
                        np.array(dequantized, copy=False).astype(np.float32),
                        expected.reshape(rows, cols),
                    )
                )

    def test_load_f8_e4m3(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        expected = [
            0,
            448,
            -448,
            -0.875,
            0.4375,
            -0.005859,
            -1.25,
            -1.25,
            -1.5,
            -0.0039,
        ]
        expected = tk.array(expected, dtype=tk.bfloat16)
        contents = b'H\x00\x00\x00\x00\x00\x00\x00{"tensor":{"dtype":"F8_E4M3","shape":[10],"data_offsets":[0,10]}}       \x00~\xfe\xb6.\x83\xba\xba\xbc\x82'
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(contents)
            f.seek(0)
            out = tk.load(f)["tensor"]
        self.assertTrue(tk.allclose(tk.from_fp8(out), expected))

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf_metadata_basic(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_tiki = os.path.join(self.test_dir, f"tiki_gguf_with_metadata.gguf")
        save_dict = {"test": tk.ones((4, 4), dtype=tk.int32)}
        metadata = {}

        # Empty works
        tk.save_gguf(save_file_tiki, save_dict, metadata)

        # Loads without the metadata
        load_dict = tk.load(save_file_tiki)
        self.assertTrue("test" in load_dict)
        self.assertTrue(tk.array_equal(load_dict["test"], save_dict["test"]))

        # Loads empty metadata
        load_dict, meta_load_dict = tk.load(save_file_tiki, return_metadata=True)
        self.assertTrue("test" in load_dict)
        self.assertTrue(tk.array_equal(load_dict["test"], save_dict["test"]))
        self.assertEqual(len(meta_load_dict), 0)

        # Loads string metadata
        metadata = {"meta": "data"}
        tk.save_gguf(save_file_tiki, save_dict, metadata)
        load_dict, meta_load_dict = tk.load(save_file_tiki, return_metadata=True)
        self.assertTrue("test" in load_dict)
        self.assertTrue(tk.array_equal(load_dict["test"], save_dict["test"]))
        self.assertEqual(len(meta_load_dict), 1)
        self.assertTrue("meta" in meta_load_dict)
        self.assertEqual(meta_load_dict["meta"], "data")

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf_metadata_arrays(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_tiki = os.path.join(self.test_dir, f"tiki_gguf_with_metadata.gguf")
        save_dict = {"test": tk.ones((4, 4), dtype=tk.int32)}

        # Test scalars and one dimensional arrays
        for t in [
            tk.uint8,
            tk.int8,
            tk.uint16,
            tk.int16,
            tk.uint32,
            tk.int32,
            tk.uint64,
            tk.int64,
            tk.float32,
        ]:
            for shape in [(), (2,)]:
                arr = tk.random.uniform(shape=shape).astype(t)
                metadata = {"meta": arr}
                tk.save_gguf(save_file_tiki, save_dict, metadata)
                _, meta_load_dict = tk.load(save_file_tiki, return_metadata=True)
                self.assertEqual(len(meta_load_dict), 1)
                self.assertTrue("meta" in meta_load_dict)
                self.assertTrue(tk.array_equal(meta_load_dict["meta"], arr))
                self.assertEqual(meta_load_dict["meta"].dtype, arr.dtype)

        for t in [tk.float16, tk.bfloat16, tk.complex64]:
            with self.assertRaises(ValueError):
                arr = tk.array(1, t)
                metadata = {"meta": arr}
                tk.save_gguf(save_file_tiki, save_dict, metadata)

    @unittest.skipIf(platform.system() == "Windows", "GGUF is disabled on Windows")
    def test_save_and_load_gguf_metadata_mixed(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_tiki = os.path.join(self.test_dir, f"tiki_gguf_with_metadata.gguf")
        save_dict = {"test": tk.ones((4, 4), dtype=tk.int32)}

        # Test string and array
        arr = tk.array(1.5)
        metadata = {"meta1": arr, "meta2": "data"}
        tk.save_gguf(save_file_tiki, save_dict, metadata)
        _, meta_load_dict = tk.load(save_file_tiki, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 2)
        self.assertTrue("meta1" in meta_load_dict)
        self.assertTrue(tk.array_equal(meta_load_dict["meta1"], arr))
        self.assertEqual(meta_load_dict["meta1"].dtype, arr.dtype)
        self.assertTrue("meta2" in meta_load_dict)
        self.assertEqual(meta_load_dict["meta2"], "data")

        # Test list of strings
        metadata = {"meta": ["data1", "data2", "data345"]}
        tk.save_gguf(save_file_tiki, save_dict, metadata)
        _, meta_load_dict = tk.load(save_file_tiki, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 1)
        self.assertEqual(meta_load_dict["meta"], metadata["meta"])

        # Test a combination of stuff
        metadata = {
            "meta1": ["data1", "data2", "data345"],
            "meta2": tk.array([1, 2, 3, 4]),
            "meta3": "data",
            "meta4": tk.array(1.5),
        }
        tk.save_gguf(save_file_tiki, save_dict, metadata)
        _, meta_load_dict = tk.load(save_file_tiki, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 4)
        for k, v in metadata.items():
            if isinstance(v, tk.array):
                self.assertTrue(tk.array_equal(meta_load_dict[k], v))
            else:
                self.assertEqual(meta_load_dict[k], v)

    def test_save_and_load_fs(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_tiki = os.path.join(
                            self.test_dir, f"tiki_{dt}_{i}_fs.npy"
                        )
                        save_file_npy = os.path.join(
                            self.test_dir, f"npy_{dt}_{i}_fs.npy"
                        )

                        save_arr = np.random.uniform(0.0, 32.0, size=shape)
                        save_arr_npy = save_arr.astype(getattr(np, dt))
                        save_arr_tiki = tk.array(save_arr_npy)

                        with open(save_file_tiki, "wb") as f:
                            tk.save(f, save_arr_tiki)

                        np.save(save_file_npy, save_arr_npy)

                        # Load array saved by tiki as tiki array
                        with open(save_file_tiki, "rb") as f:
                            load_arr_tiki_tiki = tk.load(f)
                        self.assertTrue(
                            tk.array_equal(load_arr_tiki_tiki, save_arr_tiki)
                        )

                        # Load array saved by numpy as tiki array
                        with open(save_file_npy, "rb") as f:
                            load_arr_npy_tiki = tk.load(f)
                        self.assertTrue(
                            tk.array_equal(load_arr_npy_tiki, save_arr_tiki)
                        )

                        # Load array saved by tiki as numpy array
                        load_arr_tiki_npy = np.load(save_file_tiki)
                        self.assertTrue(np.array_equal(load_arr_tiki_npy, save_arr_npy))

    def test_savez_and_loadz(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                shapes = [(6,), (6, 6), (4, 1, 3, 1, 2)]
                save_file_tiki_uncomp = os.path.join(
                    self.test_dir, f"tiki_{dt}_uncomp.npz"
                )
                save_file_npy_uncomp = os.path.join(
                    self.test_dir, f"npy_{dt}_uncomp.npz"
                )
                save_file_tiki_comp = os.path.join(self.test_dir, f"tiki_{dt}_comp.npz")
                save_file_npy_comp = os.path.join(self.test_dir, f"npy_{dt}_comp.npz")

                # Make dictionary of multiple
                save_arrs_npy = {
                    f"save_arr_{i}": np.random.uniform(
                        0.0, 32.0, size=shapes[i]
                    ).astype(getattr(np, dt))
                    for i in range(len(shapes))
                }
                save_arrs_tiki = {k: tk.array(v) for k, v in save_arrs_npy.items()}

                # Save as npz files
                np.savez(save_file_npy_uncomp, **save_arrs_npy)
                tk.savez(save_file_tiki_uncomp, **save_arrs_tiki)
                np.savez_compressed(save_file_npy_comp, **save_arrs_npy)
                tk.savez_compressed(save_file_tiki_comp, **save_arrs_tiki)

                for save_file_npy, save_file_tiki in (
                    (save_file_npy_uncomp, save_file_tiki_uncomp),
                    (save_file_npy_comp, save_file_tiki_comp),
                ):
                    # Load array saved by tiki as tiki array
                    load_arr_tiki_tiki = tk.load(save_file_tiki)
                    for k, v in load_arr_tiki_tiki.items():
                        self.assertTrue(tk.array_equal(save_arrs_tiki[k], v))

                    # Load arrays saved by numpy as tiki arrays
                    load_arr_npy_tiki = tk.load(save_file_npy)
                    for k, v in load_arr_npy_tiki.items():
                        self.assertTrue(tk.array_equal(save_arrs_tiki[k], v))

                    # Load array saved by tiki as numpy array
                    load_arr_tiki_npy = np.load(save_file_tiki)
                    for k, v in load_arr_tiki_npy.items():
                        self.assertTrue(np.array_equal(save_arrs_npy[k], v))

    def test_non_contiguous(self):
        a = tk.broadcast_to(tk.array([1, 2]), [4, 2])

        save_file = os.path.join(self.test_dir, "a.npy")
        tk.save(save_file, a)
        aload = tk.load(save_file)
        self.assertTrue(tk.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.safetensors")
        tk.save_safetensors(save_file, {"a": a})
        aload = tk.load(save_file)["a"]
        self.assertTrue(tk.array_equal(a, aload))

        if platform.system() == "Windows":
            return

        save_file = os.path.join(self.test_dir, "a.gguf")
        tk.save_gguf(save_file, {"a": a})
        aload = tk.load(save_file)["a"]
        self.assertTrue(tk.array_equal(a, aload))

        # safetensors and gguf only work with row contiguous
        # make sure col contiguous is handled properly
        save_file = os.path.join(self.test_dir, "a.safetensors")
        a = tk.arange(4).reshape(2, 2).T
        tk.save_safetensors(save_file, {"a": a})
        aload = tk.load(save_file)["a"]
        self.assertTrue(tk.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.gguf")
        tk.save_gguf(save_file, {"a": a})
        aload = tk.load(save_file)["a"]
        self.assertTrue(tk.array_equal(a, aload))

    def test_load_donation(self):
        x = tk.random.normal((1024,))
        tk.eval(x)
        save_file = os.path.join(self.test_dir, "donation.npy")
        tk.save(save_file, x)
        tk.synchronize()

        tk.reset_peak_memory()
        scale = tk.array(2.0)
        y = tk.load(save_file)
        tk.eval(y)
        tk.synchronize()
        load_only = tk.get_peak_memory()
        y = tk.load(save_file) * scale
        tk.eval(y)
        tk.synchronize()
        load_with_binary = tk.get_peak_memory()

        self.assertEqual(load_only, load_with_binary)

    def test_save_and_load_empty(self):
        for i, shape in enumerate([(0,), (0, 3), (3, 0), (2, 0, 4)]):
            with self.subTest(shape=shape):
                save_arr = tk.zeros(shape)

                npy_file = os.path.join(self.test_dir, f"empty_{i}.npy")
                tk.save(npy_file, save_arr)
                self.assertEqual(tk.load(npy_file).shape, shape)
                # numpy can read what we wrote
                self.assertEqual(np.load(npy_file).shape, shape)

                st_file = os.path.join(self.test_dir, f"empty_{i}.safetensors")
                tk.save_safetensors(st_file, {"x": save_arr})
                self.assertEqual(tk.load(st_file)["x"].shape, shape)

        # An empty array alongside a normal one round trips both
        npz_file = os.path.join(self.test_dir, "empty.npz")
        tk.savez(npz_file, x=tk.zeros((0, 3)), y=tk.ones((2, 2)))
        loaded = tk.load(npz_file)
        self.assertEqual(loaded["x"].shape, (0, 3))
        self.assertTrue(tk.array_equal(loaded["y"], tk.ones((2, 2))))

        st_file = os.path.join(self.test_dir, "empty_mixed.safetensors")
        tk.save_safetensors(st_file, {"x": tk.zeros((0, 3)), "y": tk.ones((2, 2))})
        loaded = tk.load(st_file)
        self.assertEqual(loaded["x"].shape, (0, 3))
        self.assertTrue(tk.array_equal(loaded["y"], tk.ones((2, 2))))


if __name__ == "__main__":
    tiki_tests.TIKITestRunner()
