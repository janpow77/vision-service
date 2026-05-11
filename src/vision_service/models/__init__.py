"""Modell-Loader fuer die einzelnen Vision/OCR-Backends.

Jedes Submodul kapselt **ein** Backend hinter einer einheitlichen Klasse mit
``available()`` und ``run(...)``. Die Registry in :mod:`vision_service.registry`
haelt Singletons, damit Modelle nur einmal geladen werden.
"""
