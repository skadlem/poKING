import pytest

from pokr.connector import available_plugins, build_strategy, register_plugin
from pokr.opponents import CallingStation


def test_register_and_build():
    register_plugin("testbot", lambda: CallingStation())
    s = build_strategy("testbot")
    assert isinstance(s, CallingStation)


def test_available_lists_registered():
    register_plugin("testbot2", lambda: CallingStation())
    assert "testbot2" in available_plugins()


def test_unknown_raises():
    with pytest.raises(KeyError):
        build_strategy("does_not_exist")
