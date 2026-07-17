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
        assert lb.hosts == ["http://vllm:8000/v1", "https://ollama.com"]


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
        for _ in range(10):
            ep = lb.acquire()
            assert ep is eps[0]
            lb.release(ep, success=True)

    def test_fails_over_on_error_and_sticks_to_alt(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        lb.release(lb.acquire(), success=False)  # server1 errors/times out
        for _ in range(10):
            ep = lb.acquire()
            assert ep is eps[1]
            lb.release(ep, success=True)

    def test_no_switch_back_after_failed_endpoint_recovers(self, monkeypatch):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        lb.release(lb.acquire(), success=False)
        failover = lb.acquire()  # moves to server2
        assert failover is eps[1]
        lb.release(failover, success=True)
        # Cooldown of 0 means server1 is immediately healthy again, but
        # inference must stay on server2 for locality.
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        for _ in range(10):
            ep = lb.acquire()
            assert ep is eps[1]
            lb.release(ep, success=True)

    def test_all_unhealthy_stays_on_current(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        for e in eps:
            lb.release(e, success=False)
        assert lb.acquire() is eps[0]

    def test_concurrent_requests_share_the_sticky_endpoint(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        first = lb.acquire()
        second = lb.acquire()  # first still in flight — same host anyway
        assert first is second is eps[0]

    def test_new_sessions_assigned_round_robin(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        assert lb.acquire(key="a") is eps[0]
        assert lb.acquire(key="b") is eps[1]
        assert lb.acquire(key="c") is eps[0]

    def test_each_session_sticks_to_its_host(self):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        lb.release(lb.acquire(key="a"), success=True)  # a → server1
        lb.release(lb.acquire(key="b"), success=True)  # b → server2
        for _ in range(5):
            a = lb.acquire(key="a")
            b = lb.acquire(key="b")
            assert a is eps[0]
            assert b is eps[1]
            lb.release(a, success=True)
            lb.release(b, success=True)

    def test_failover_moves_only_affected_sessions(self, monkeypatch):
        eps = _two_endpoints()
        lb = LoadBalancer(eps, algorithm="sticky")
        a = lb.acquire(key="a")  # a → server1
        lb.release(a, success=False)  # server1 errors/times out
        lb.release(lb.acquire(key="b"), success=True)  # b → server2
        assert lb.acquire(key="a") is eps[1]  # a failed over
        # server1 recovers, but a stays on server2 for locality; new
        # sessions can be assigned to server1 again.
        monkeypatch.setattr(balancer_mod, "FAILURE_COOLDOWN", 0.0)
        assert lb.acquire(key="a") is eps[1]
        assert lb.acquire(key="c") is eps[0]


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
