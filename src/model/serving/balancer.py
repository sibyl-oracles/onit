"""
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Load balancer for multiple model-serving endpoints.

Distributes chat requests across two (or more) LLM servers — e.g. two vLLM
instances, one vLLM + one Ollama cloud, or two Ollama cloud endpoints.
Provider-specific behavior (API key resolution, client type) is handled
downstream by chat(), which auto-detects the provider from each host URL.

All algorithms double as automatic failover: an endpoint that produced a
failed request is put on a cooldown and skipped while any healthy endpoint
remains available.

Ollama endpoints (cloud or local) are fallback-only: whenever at least one
non-Ollama endpoint (vLLM, OpenRouter, ...) is healthy, Ollama endpoints are
kept out of rotation and only serve requests while every non-Ollama endpoint
is cooling down.
"""

import logging
import random
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Seconds an endpoint stays deprioritized after a failed request before it
# re-enters rotation. Keeps a downed server out of the way without ever
# removing it permanently.
FAILURE_COOLDOWN = 60.0

ALGORITHMS = ("sticky", "round_robin", "random", "least_busy")
DEFAULT_ALGORITHM = "sticky"

# Cap on remembered sticky assignments; oldest sessions are evicted first.
MAX_STICKY_KEYS = 1024


def _is_ollama_host(host: str) -> bool:
    """True for Ollama endpoints: cloud (ollama.com/.ai) or local (:11434)."""
    return "ollama.com" in host or "ollama.ai" in host or ":11434" in host


@dataclass
class ServerEndpoint:
    """One model server: host URL plus its API key and optional model name."""
    host: str
    host_key: str = "EMPTY"
    model: str | None = None
    name: str = ""
    # Runtime state managed by LoadBalancer
    in_flight: int = field(default=0, repr=False)
    failed_at: float = field(default=0.0, repr=False)

    @property
    def is_ollama(self) -> bool:
        return _is_ollama_host(self.host)

    def is_healthy(self, now: float | None = None) -> bool:
        if not self.failed_at:
            return True
        now = time.monotonic() if now is None else now
        return (now - self.failed_at) >= FAILURE_COOLDOWN


class LoadBalancer:
    """Selects a ServerEndpoint per request using a configurable algorithm.

    Algorithms:
      - sticky:      each session's first request is assigned an endpoint at
                     random (so load spreads across hosts), then all of that
                     session's inference stays on the same endpoint for
                     locality; a timeout/error fails it over to the alternate,
                     where it stays (default)
      - round_robin: alternate between endpoints in order
      - random:      pick a random endpoint
      - least_busy:  pick the endpoint with the fewest in-flight requests

    Usage:
        endpoint = balancer.acquire()
        try:
            ... send request to endpoint.host ...
        finally:
            balancer.release(endpoint, success=request_succeeded)
    """

    def __init__(self, endpoints: list[ServerEndpoint],
                 algorithm: str = DEFAULT_ALGORITHM):
        if not endpoints:
            raise ValueError("LoadBalancer requires at least one endpoint")
        if algorithm not in ALGORITHMS:
            logger.warning(
                "Unknown load balancing algorithm %r, falling back to %r "
                "(valid: %s)", algorithm, DEFAULT_ALGORITHM, ", ".join(ALGORITHMS))
            algorithm = DEFAULT_ALGORITHM
        self.endpoints = list(endpoints)
        self.algorithm = algorithm
        self._lock = threading.Lock()
        self._rr_index = 0
        # sticky: session key → index of its assigned endpoint
        self._sticky_map: dict[str, int] = {}

    @property
    def hosts(self) -> list[str]:
        return [ep.host for ep in self.endpoints]

    def _rr_pick(self, candidates: list[ServerEndpoint]) -> ServerEndpoint:
        """Advance the round-robin cursor and land on a candidate.

        Walks the full endpoint list so the cycle order stays stable even
        while some endpoints are cooling down.
        """
        for _ in range(len(self.endpoints)):
            ep = self.endpoints[self._rr_index % len(self.endpoints)]
            self._rr_index += 1
            if ep in candidates:
                return ep
        return candidates[0]

    def acquire(self, key: str | None = None) -> ServerEndpoint:
        """Pick an endpoint for the next request and mark it in-flight.

        ``key`` identifies the client/session for the sticky algorithm: a
        session's first request is assigned an endpoint round-robin, and every
        later request with the same key stays on that endpoint for locality.
        Other algorithms ignore ``key``.

        Unhealthy endpoints (recent failure, still in cooldown) are skipped
        unless every endpoint is unhealthy, in which case all are considered
        so the caller can still make progress.

        Ollama endpoints are fallback-only: while any healthy non-Ollama
        endpoint exists, Ollama ones are excluded from the candidates. A
        sticky session that failed over to Ollama therefore moves back to a
        vLLM/OpenRouter endpoint as soon as one recovers.
        """
        with self._lock:
            now = time.monotonic()
            candidates = [ep for ep in self.endpoints if ep.is_healthy(now)]
            if not candidates:
                candidates = self.endpoints
            preferred = [ep for ep in candidates if not ep.is_ollama]
            if preferred:
                candidates = preferred

            if self.algorithm == "sticky":
                # Locality: keep a session's inference on its assigned
                # endpoint. Only a timeout/error (cooldown) moves it to the
                # next healthy endpoint, which then becomes the session's new
                # sticky target — no switching back when the failed one
                # recovers (except back to a recovered non-Ollama endpoint,
                # since Ollama is fallback-only). First-time keys are
                # assigned a random candidate so load spreads across hosts.
                key = key or ""
                idx = self._sticky_map.get(key)
                if idx is not None and self.endpoints[idx] in candidates:
                    chosen = self.endpoints[idx]
                else:
                    chosen = random.choice(candidates)
                    if idx is not None:
                        logger.warning("Sticky failover (%s) → %s",
                                       key or "default",
                                       chosen.name or chosen.host)
                    self._sticky_map.pop(key, None)
                    self._sticky_map[key] = self.endpoints.index(chosen)
                    while len(self._sticky_map) > MAX_STICKY_KEYS:
                        self._sticky_map.pop(next(iter(self._sticky_map)))
            elif len(candidates) == 1:
                chosen = candidates[0]
            elif self.algorithm == "random":
                chosen = random.choice(candidates)
            elif self.algorithm == "least_busy":
                chosen = min(candidates, key=lambda ep: ep.in_flight)
            else:  # round_robin
                chosen = self._rr_pick(candidates)

            chosen.in_flight += 1
            if len(self.endpoints) > 1:
                logger.info("Load balancer (%s) → %s", self.algorithm,
                            chosen.name or chosen.host)
            return chosen

    def release(self, endpoint: ServerEndpoint, success: bool) -> None:
        """Mark a request finished; on failure start the endpoint's cooldown."""
        with self._lock:
            endpoint.in_flight = max(0, endpoint.in_flight - 1)
            if success:
                endpoint.failed_at = 0.0
            else:
                endpoint.failed_at = time.monotonic()
                if len(self.endpoints) > 1:
                    logger.warning(
                        "Endpoint %s failed; cooling down for %.0fs",
                        endpoint.name or endpoint.host, FAILURE_COOLDOWN)
