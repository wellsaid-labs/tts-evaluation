# Test Data

This directory is for writing temporary files or storing files. Mostly the namespace is dictated by
the `tests/` directory. For example `tests/_test_data/bin` is used by `tests/bin`.
The `_disk` directory is special and is related to the top-level `disk` directory.

TODO: At the moment, we don't support sharing files across tests. We should consider allowing
this for commonly needed test files.
