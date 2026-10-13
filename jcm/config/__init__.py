"""Marks ``jcm/config`` as a real package so the packaged Hydra config tree is
reliably reachable from a downstream app via ``hydra.searchpath: [pkg://jcm.config]``.

Hydra's ``pkg://`` provider only reports a directory as *available* when it is a
regular package (its availability check looks for this ``__init__.py``). Without
it ``jcm.config`` is a namespace package: Hydra still reads it today but warns
``provider=hydra.searchpath ... is not available`` and a stricter release could
drop it. The ``__init__.py`` makes the searchpath contract in #757 solid and
warning-free. See ``docs/source/design/packaged_config_tree.md``.
"""
