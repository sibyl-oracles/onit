"""Tests for src/model/serving/balancer.py — LoadBalancer and ServerEndpoint."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.serving import balancer as balancer_mod
from model.serving.balancer import LoadBalancer, ServerEndpoint


def _two_endpoints():
    return [
        ServerEndpoint(host="http://vllm:8000/v1", name="server1"),
        ServerEndpoint(host="http://vllm2:8000/v1", name="server2"),
    ]


def _vllm_and_ollama():
    return [
        ServerEndpoint(host="http://vllm:8000/v1", name="server1"),
        ServerEndpoint(host="https://ollama.com", host_key="key2",
                       model="qwen3", name="server2"),
    ]


class TestLoadBalancerConstruction:
    def test_requires_endpoints(self):
        with pytest.raises(ValueError):
            LoadBalancer([])

    def test_unknown_algorithm_falls_back_to_default(self):
        lb = LoadBalancer(_two_endpoints(), algorithm="bogus")
        assert lb.algorithm == balancer_mod.DEFAULT_ALGORITHM

    def test_default_algorithm_is_sticky(self):
        assert LoadBalancer(_two_endpoints()).algorithm == "sticky"

    def test_hosts_property(self):
        lb = LoadBalancer(_two_endpoints())
        assert lb.hosts == ["http://vllm:8000/v1", "http://vllm2:8000/v1"]


class TestSingleEndpoint:
    def test_always_returns_the_only_endpoint(self):
        ep = ServerEndpoint(host="http://vllm:8000/v1")
        lb = LoadBalancer([ep])
        for _ in range(3):
            got = lb.acquire()
            assert got is ep
            lb.release(got, success=True)

    def test_unhealthy_single_endpoint_still_returned(self):
        ep = ServerEndpoint(host="http://vllm:8000/v1")
        lb = LoadBalancer([ep])
        lb.release(lb.acquire(), success=False)
        assert lb.acquire() is ep


class TestSticky:
    def test_stays_on_one_endpoint(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        first = lb.acquire()
        assert first in eps
        lb.release(first, success=True)
        for _ in range(10):
            ep = lb.acquire()
            assert ep is first
            lb.release(ep, success=True)

    def test_fails_over_on_error_and_sticks_to_alt(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        first = lb.acquire()
        lb.release(first, success=False)  # errors/times out
        alt = next(e for e in eps if e is not first)
        for _ in range(10):
            ep = lb.acquire()
            assert ep is alt
            lb.release(ep, success=True)

    def test_no_switch_back_after_failed_endpoint_recovers(self, monkeypatch):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        first = lb.acquire()
        lb.release(first, success=False)
        failover = lb.acquire()  # moves to the alternate
        assert failover is not first
        lb.release(failover, success=True)
        # Cooldown of 0 means the failed endpoint is immediately healthy
        # again, but inference must stay on the failover target for locality.
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        for _ in range(10):
            ep = lb.acquire()
            assert ep is failover
            lb.release(ep, success=True)

    def test_all_unhealthy_still_serves_and_sticks(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        for e in eps:
            lb.release(e, success=False)
        first = lb.acquire()
        assert first in eps
        assert lb.acquire() is first

    def test_concurrent_requests_share_the_sticky_endpoint(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        first = lb.acquire()
        second = lb.acquire()  # first still in flight — same host anyway
        assert first is second
        assert first in eps

    def test_new_sessions_assigned_randomly_across_hosts(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        seen = set()
        for i in range(100):
            ep = lb.acquire(key=f"session-{i}")
            seen.add(ep.name)
            lb.release(ep, success=True)
        assert seen == {"server1", "server2"}

    def test_each_session_sticks_to_its_host(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        a_ep = lb.acquire(key="a")
        lb.release(a_ep, success=True)
        b_ep = lb.acquire(key="b")
        lb.release(b_ep, success=True)
        for _ in range(5):
            a = lb.acquire(key="a")
            b = lb.acquire(key="b")
            assert a is a_ep
            assert b is b_ep
            lb.release(a, success=True)
            lb.release(b, success=True)

    def test_failover_moves_only_affected_sessions(self, monkeypatch):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        a = lb.acquire(key="a")
        lb.release(a, success=False)  # a's endpoint errors/times out
        alt = next(e for e in eps if e is not a)
        lb.release(lb.acquire(key="b"), success=True)  # b → healthy host
        assert lb.acquire(key="a") is alt  # a failed over
        # a's original endpoint recovers, but a stays on the failover
        # target for locality; new sessions may use either host again.
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        assert lb.acquire(key="a") is alt
        assert lb.acquire(key="c") in eps


class TestRoundRobin:
    def test_alternates_between_endpoints(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="round_robin")
        picks = []
        for _ in range(4):
            ep = lb.acquire()
            picks.append(ep.name)
            lb.release(ep, success=True)
        assert picks == ["server1", "server2", "server1", "server2"]


class TestRandom:
    def test_uses_both_endpoints_eventually(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="random")
        seen = set()
        for _ in range(50):
            ep = lb.acquire()
            seen.add(ep.name)
            lb.release(ep, success=True)
        assert seen == {"server1", "server2"}


class TestLeastBusy:
    def test_picks_endpoint_with_fewest_in_flight(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="least_busy")
        first = lb.acquire()   # both idle → picks one
        second = lb.acquire()  # first is busy → must pick the other
        assert {first.name, second.name} == {"server1", "server2"}
        lb.release(first, success=True)
        third = lb.acquire()   # first is now idle again
        assert third is first

    def test_in_flight_never_goes_negative(self):
        ep = ServerEndpoint(host="http://vllm:8000/v1")
        lb = LoadBalancer([ep])
        lb.release(ep, success=True)
        assert ep.in_flight == 0


class TestFailover:
    def test_failed_endpoint_skipped_during_cooldown(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="round_robin")
        ep = lb.acquire()
        lb.release(ep, success=False)
        # While cooling down, only the other endpoint is used
        for _ in range(4):
            got = lb.acquire()
            assert got is not ep
            lb.release(got, success=True)

    def test_failed_endpoint_returns_after_cooldown(self, monkeypatch):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="round_robin")
        ep = lb.acquire()
        lb.release(ep, success=False)
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        names = set()
        for _ in range(4):
            got = lb.acquire()
            names.add(got.name)
            lb.release(got, success=True)
        assert names == {"server1", "server2"}

    def test_success_clears_failure_state(self):
        ep = ServerEndpoint(host="http://vllm:8000/v1")
        lb = LoadBalancer([ep])
        lb.release(lb.acquire(), success=False)
        assert not ep.is_healthy()
        lb.release(lb.acquire(), success=True)
        assert ep.is_healthy()

    def test_all_unhealthy_still_serves(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="least_busy")
        for e in eps:
            lb.release(e, success=False)
        assert lb.acquire() in eps


class TestOllamaFallback:
    def test_is_ollama_detection(self):
        assert ServerEndpoint(host="https://ollama.com").is_ollama
        assert ServerEndpoint(host="https://api.ollama.ai/v1").is_ollama
        assert ServerEndpoint(host="http://localhost:11434/v1").is_ollama
        assert not ServerEndpoint(host="http://vllm:8000/v1").is_ollama
        assert not ServerEndpoint(host="https://openrouter.ai/api/v1").is_ollama

    def test_new_sessions_always_prefer_vllm(self):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="sticky")
        for i in range(20):
            ep = lb.acquire(key=f"session-{i}")
            assert ep is eps[0]
            lb.release(ep, success=True)

    def test_round_robin_skips_ollama_while_vllm_healthy(self):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="round_robin")
        for _ in range(6):
            ep = lb.acquire()
            assert ep is eps[0]
            lb.release(ep, success=True)

    def test_ollama_serves_while_vllm_cooling_down(self):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="sticky")
        lb.release(lb.acquire(), success=False)  # vLLM errors/times out
        assert lb.acquire() is eps[1]

    def test_session_returns_to_vllm_after_recovery(self, monkeypatch):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="sticky")
        lb.release(lb.acquire(key="a"), success=False)  # vLLM fails
        failover = lb.acquire(key="a")  # a → ollama fallback
        assert failover is eps[1]
        lb.release(failover, success=True)
        # vLLM comes back — the session leaves the fallback immediately.
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        assert lb.acquire(key="a") is eps[0]

    def test_ollama_only_setup_uses_ollama(self):
        ep = ServerEndpoint(host="https://ollama.com", model="qwen3")
        lb = LoadBalancer([ep])
        assert lb.acquire() is ep


class TestOllamaFallbackDisabled:
    """ollama_fallback_only=False puts Ollama endpoints in normal rotation."""

    def test_default_is_fallback_only(self):
        assert LoadBalancer(_vllm_and_ollama()).ollama_fallback_only is True

    def test_round_robin_includes_ollama(self):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="round_robin",
                          ollama_fallback_only=False)
        seen = set()
        for _ in range(4):
            ep = lb.acquire()
            seen.add(ep.name)
            lb.release(ep, success=True)
        assert seen == {"server1", "server2"}

    def test_sticky_sessions_spread_across_both(self):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="sticky", ollama_fallback_only=False)
        seen = set()
        for i in range(50):
            ep = lb.acquire(key=f"session-{i}")
            seen.add(ep.name)
            lb.release(ep, success=True)
        assert seen == {"server1", "server2"}

    def test_sticky_session_stays_put_after_recovery(self, monkeypatch):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="sticky", ollama_fallback_only=False)
        first = lb.acquire(key="a")
        lb.release(first, success=False)
        failover = lb.acquire(key="a")
        assert failover is not first
        lb.release(failover, success=True)
        # The failed endpoint recovers, but with Ollama first-class there is
        # no forced move back — even when the failover target is Ollama.
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        assert lb.acquire(key="a") is failover

    def test_cooldown_failover_still_works(self):
        eps = _vllm_and_ollama()
        lb = LoadBalancer(eps, algorithm="round_robin",
                          ollama_fallback_only=False)
        lb.release(lb.acquire(), success=False)  # whichever was picked fails
        failed = [ep for ep in eps if ep.failed_at][0]
        other = eps[0] if failed is eps[1] else eps[1]
        assert lb.acquire() is other


def _tiered():
    """Two priority-1 vLLM servers with a priority-2 Ollama fallback."""
    return [
        ServerEndpoint(host="http://vllm:8000/v1", name="a", priority=1),
        ServerEndpoint(host="http://vllm2:8000/v1", name="b", priority=1),
        ServerEndpoint(host="https://ollama.com", name="c", priority=2),
    ]


class TestPriority:
    def test_default_priority_is_not_in_use(self):
        assert LoadBalancer(_two_endpoints()).uses_priority is False

    def test_any_explicit_priority_enables_tiers(self):
        eps = _two_endpoints()
        eps[1].priority = 1
        assert LoadBalancer(eps).uses_priority is True

    def test_top_tier_serves_while_healthy(self):
        eps = _tiered()
        lb = LoadBalancer(eps, algorithm="round_robin")
        for _ in range(6):
            ep = lb.acquire()
            assert ep.priority == 1
            lb.release(ep, success=True)

    def test_load_balances_within_a_tier(self):
        lb = LoadBalancer(_tiered(), algorithm="round_robin")
        seen = set()
        for _ in range(6):
            ep = lb.acquire()
            seen.add(ep.name)
            lb.release(ep, success=True)
        assert seen == {"a", "b"}

    def test_lower_tier_serves_only_when_top_tier_is_down(self):
        eps = _tiered()
        lb = LoadBalancer(eps, algorithm="round_robin")
        for ep in eps[:2]:
            lb.release(ep, success=False)
        assert lb.acquire() is eps[2]

    def test_partial_tier_outage_stays_in_tier(self):
        eps = _tiered()
        lb = LoadBalancer(eps, algorithm="round_robin")
        lb.release(eps[0], success=False)  # only one of the two fails
        for _ in range(4):
            ep = lb.acquire()
            assert ep is eps[1]
            lb.release(ep, success=True)

    def test_all_unhealthy_falls_back_to_best_tier(self):
        eps = _tiered()
        lb = LoadBalancer(eps, algorithm="round_robin")
        for ep in eps:
            lb.release(ep, success=False)
        # Nothing is healthy, so every endpoint is a candidate again — but
        # the ranking still applies and keeps traffic on the preferred tier.
        assert lb.acquire().priority == 1

    def test_sticky_returns_to_top_tier_after_recovery(self, monkeypatch):
        eps = _tiered()
        lb = LoadBalancer(eps, algorithm="sticky")
        for ep in eps[:2]:
            lb.release(ep, success=False)
        assert lb.acquire(key="a") is eps[2]  # failed over to the fallback
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        assert lb.acquire(key="a").priority == 1

    def test_negative_and_sparse_numbers_rank_normally(self):
        eps = [
            ServerEndpoint(host="http://a/v1", name="a", priority=5),
            ServerEndpoint(host="http://b/v1", name="b", priority=-3),
        ]
        lb = LoadBalancer(eps, algorithm="round_robin")
        assert lb.acquire() is eps[1]

    def test_priority_overrides_ollama_fallback_only(self):
        """An explicit ranking beats the implicit "Ollama last" default."""
        eps = [
            ServerEndpoint(host="https://ollama.com", name="ollama",
                           priority=1),
            ServerEndpoint(host="http://vllm:8000/v1", name="vllm",
                           priority=2),
        ]
        lb = LoadBalancer(eps, algorithm="round_robin",
                          ollama_fallback_only=True)
        for _ in range(4):
            ep = lb.acquire()
            assert ep.name == "ollama"
            lb.release(ep, success=True)

    def test_least_busy_respects_tiers(self):
        eps = _tiered()
        lb = LoadBalancer(eps, algorithm="least_busy")
        first = lb.acquire()
        second = lb.acquire()
        third = lb.acquire()  # both tier-1 servers busy — no spillover
        assert {first.name, second.name} == {"a", "b"}
        assert third.priority == 1

    def test_preferred_endpoint(self):
        assert LoadBalancer(_tiered()).preferred.name == "a"
        assert LoadBalancer(_two_endpoints()).preferred.name == "server1"
        # Without priorities, the fallback rule picks the representative.
        assert LoadBalancer(_vllm_and_ollama()).preferred.name == "server1"

    def test_describe_groups_by_tier(self):
        assert LoadBalancer(_tiered()).describe() == "1: a, b | 2: c"
        assert LoadBalancer(_two_endpoints()).describe() == (
            "http://vllm:8000/v1, http://vllm2:8000/v1")
