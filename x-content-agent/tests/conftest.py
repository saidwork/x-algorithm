"""Pytest configuration for async tests without external plugins."""

import asyncio
import inspect

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
        if name in pyfuncitem.funcargs
    }
    asyncio.run(test_func(**kwargs))
    # Some sync tests call asyncio.get_event_loop(); keep one available.
    asyncio.set_event_loop(asyncio.new_event_loop())
    return True
