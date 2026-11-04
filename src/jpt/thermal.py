"""Thermal state of the Mac.

Reads ``NSProcessInfo.processInfo.thermalState`` through the Objective-C runtime:
0 nominal, 1 fair, 2 serious, 3 critical. On a fanless MacBook Air, sustained
inference pushes it up and the chip slows down to cool off: the throttling shows
up as a falling tokens/s curve in ``jpt thermal``.
"""

from __future__ import annotations

import ctypes
import ctypes.util

LABELS = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}

_objc = None


def thermal_state() -> int | None:
    global _objc
    try:
        if _objc is None:
            ctypes.cdll.LoadLibrary(ctypes.util.find_library("Foundation"))
            _objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
            _objc.objc_getClass.restype = ctypes.c_void_p
            _objc.objc_getClass.argtypes = [ctypes.c_char_p]
            _objc.sel_registerName.restype = ctypes.c_void_p
            _objc.sel_registerName.argtypes = [ctypes.c_char_p]
        send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", _objc))
        send_long = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", _objc))
        cls = _objc.objc_getClass(b"NSProcessInfo")
        info = send(cls, _objc.sel_registerName(b"processInfo"))
        return int(send_long(info, _objc.sel_registerName(b"thermalState")))
    except Exception:
        return None


def label(state: int | None) -> str:
    return LABELS.get(state, "unknown") if state is not None else "unknown"
