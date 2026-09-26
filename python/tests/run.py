import os
import sys

# Use regular fp32 precision for tests
os.environ["TIKI_ENABLE_TF32"] = "0"

# Do not abort on cache thrashing
os.environ["TIKI_ENABLE_CACHE_THRASHING_CHECK"] = "0"

__unittest = True

import tiki_tests

if __name__ == "__main__":
    # Run all tests by default.
    dirname = os.path.dirname(os.path.realpath(__file__))
    argv = [sys.argv[0], "discover", dirname, *sys.argv[1:]]
    tiki_tests.TIKITestRunner(argv=argv, module=None)
