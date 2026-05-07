"""Reactor environments and physics for the HolosGen microreactor.

The Gym/PettingZoo env classes pull in heavy dependencies (gymnasium,
pettingzoo). They're loaded lazily so that lighter consumers — like
`nmpc.ReactorModel`, which only needs `envs.holos_constants` — don't
have to install or import the RL stack.
"""
import importlib

__all__ = ["HolosPK", "HolosMulti", "HolosSingle", "HolosMARL"]

_LAZY = {
    "HolosPK": "holos_pk",
    "HolosMulti": "holos_multi",
    "HolosSingle": "holos_single",
    "HolosMARL": "holos_marl",
}


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
