import sys
import pytest

import dpeva.cli as cli


def test_cli_dispatch_train_without_banner(monkeypatch):
    called = {"train": False}

    def fake_train(args):
        called["train"] = True
        assert args.config == "config.json"

    monkeypatch.setattr(cli, "handle_train", fake_train)
    monkeypatch.setattr(cli, "show_banner", lambda: (_ for _ in ()).throw(AssertionError("banner should not be called")))
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "train", "config.json"])

    cli.main()
    assert called["train"] is True


def test_cli_exit_on_handler_error(monkeypatch):
    def fake_train(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "handle_train", fake_train)
    monkeypatch.setattr(cli, "show_banner", lambda: None)
    monkeypatch.setattr(sys, "argv", ["dpeva", "--no-banner", "train", "config.json"])

    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
