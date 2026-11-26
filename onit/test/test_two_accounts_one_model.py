"""Tests for running two accounts on one model server.

The regression this suite pins: adding a second Ollama account — same URL,
same model, distinct account label — must keep both keys instead of the
second overwriting the first.  The label is the only thing that tells the
two endpoints apart, so identity, key storage, and the runtime's endpoint
reader all have to honor it.
"""

import pytest

from onit.model.serving.balancer import LoadBalancer, ServerEndpoint
from onit.onit import OnIt
from onit import setup as setup_mod


class TestLabeledEndpointIdentity:
    def test_two_accounts_on_one_model_are_distinct(self):
        plain = setup_mod.endpoint_identity("https://ollama.com", "glm-5.3:cloud")
        labeled = setup_mod.endpoint_identity("https://ollama.com", "glm-5.3:cloud", "work")
        assert plain != labeled
        other = setup_mod.endpoint_identity("https://ollama.com", "glm-5.3:cloud", "home")
        assert labeled != other

    def test_two_unlabeled_copies_stay_one_identity(self):
        """A label exists to name an account; duplicates without one remain
        a repeat, and the wizard still refuses them as before."""
        assert setup_mod.endpoint_identity(
            "https://ollama.com", "m"
        ) == setup_mod.endpoint_identity("https://ollama.com", "m")

    def test_entry_identity_carries_the_name(self):
        assert setup_mod.entry_identity(
            {"host": "https://ollama.com", "model": "m", "name": "work"}
        ) == ("https://ollama.com", "m", "work")


class TestLabeledKeyStorage:
    @pytest.fixture(autouse=True)
    def _no_keychain(self, monkeypatch):
        self.store = {}
        monkeypatch.setattr(setup_mod, "get_secret", lambda key: self.store.get(key))
        monkeypatch.setattr(
            setup_mod, "store_secret", lambda key, value: self.store.__setitem__(key, value)
        )
        monkeypatch.setattr(setup_mod, "delete_secret", lambda key: self.store.pop(key, None))
        for var in list(os_environ_like()):
            if var.startswith("ONIT_ENDPOINT_KEY_"):
                del os_environ_like()[var]

    def test_each_account_stores_and_reads_its_own_key(self):
        setup_mod.store_endpoint_key("https://ollama.com", "sk-a")
        setup_mod.store_endpoint_key("https://ollama.com", "sk-b", "work")
        assert setup_mod.get_endpoint_key("https://ollama.com") == "sk-a"
        assert setup_mod.get_endpoint_key("https://ollama.com", "work") == "sk-b"

    def test_deleting_one_account_leaves_the_other(self):
        setup_mod.store_endpoint_key("https://ollama.com", "sk-a")
        setup_mod.store_endpoint_key("https://ollama.com", "sk-b", "work")
        setup_mod.delete_endpoint_key("https://ollama.com", "work")
        assert setup_mod.get_endpoint_key("https://ollama.com") == "sk-a"
        # The labeled lookup falls back to the plain slot once its own key
        # is gone — the entry degrades to the first account, not to nothing.


class TestRuntimeKeepsBothAccounts:
    def test_the_balancer_holds_two_labeled_endpoints(self):
        """The endpoint reader's duplicate filter must not drop a second
        account: same host, same model, distinct label."""
        endpoints = OnIt._parse_endpoint_list(
            [
                {"host": "https://ollama.com", "model": "glm-5.3:cloud", "name": "work"},
                {"host": "https://ollama.com", "model": "glm-5.3:cloud", "name": "home"},
            ]
        )
        assert len(endpoints) == 2
        assert [ep.name for ep in endpoints] == ["work", "home"]

    def test_a_genuine_duplicate_is_still_dropped(self):
        endpoints = OnIt._parse_endpoint_list(
            [
                {"host": "https://ollama.com", "model": "glm-5.3:cloud"},
                {"host": "https://ollama.com", "model": "glm-5.3:cloud"},
            ]
        )
        assert len(endpoints) == 1

    def test_the_balancer_load_balances_across_the_two(self):
        lb = LoadBalancer(
            [
                ServerEndpoint(host="https://ollama.com", model="glm-5.3:cloud", name="work"),
                ServerEndpoint(host="https://ollama.com", model="glm-5.3:cloud", name="home"),
            ],
            "round_robin",
        )
        picked = {lb.acquire().name for _ in range(4)}
        assert picked == {"work", "home"}


def os_environ_like():
    import os

    return os.environ
