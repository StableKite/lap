# StableKite free-threaded build

This fork keeps the LAPJV/LAPMOD API and algorithms. It requires Cython 3.1+
to emit CPython's free-threading module declaration. It does not force runtime
GIL settings. The minimum Python version follows Cython 3.1 (Python 3.8).

The C++ solver's working arrays and allocations are local to each call; there
is no mutable process-global solver state. The Cython wrapper now takes owned,
contiguous copies of native input arrays, so each solve retains its buffers
while independent calls execute concurrently. As with NumPy generally, callers
must not resize or mutate an input concurrently while it is being copied.
The extension does not expose borrowed result buffers.

`lap/tests/test_freethreading.py` checks a normal import on a free-threaded
interpreter and concurrent dense/sparse calls with independent inputs. The
existing upstream tests remain applicable. A passing concurrency test is not a
proof against every possible race or allocation failure.
