from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--online",
        action="store_true",
        default=False,
        help="run tests that exercise real external services",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "online: exercises real external services")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--online"):
        return

    skip_online = pytest.mark.skip(reason="pass --online to run tests against real external services")
    for item in items:
        if "online" in item.keywords:
            item.add_marker(skip_online)
