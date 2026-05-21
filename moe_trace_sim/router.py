from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, pstdev

from .config import ModelConfig, RoutingConfig


@dataclass
class RoutingPlan:
    assignments: list[list[int]]
    expert_counts: list[int]
    load_imbalance: float
    cv: float
    empty_expert_ratio: float

    @property
    def total_routed_tokens(self) -> int:
        return sum(self.expert_counts)


def route_tokens(model: ModelConfig, routing: RoutingConfig) -> RoutingPlan:
    rng = random.Random(routing.seed)
    weights = _weights(model.num_experts, routing)
    assignments: list[list[int]] = []
    counts = [0 for _ in range(model.num_experts)]

    for _ in range(model.num_tokens):
        experts = _weighted_sample_without_replacement(
            rng, model.num_experts, model.top_k, weights
        )
        assignments.append(experts)
        for expert in experts:
            counts[expert] += 1

    avg = mean(counts) if counts else 0.0
    sd = pstdev(counts) if len(counts) > 1 else 0.0
    empty = sum(1 for c in counts if c == 0)
    return RoutingPlan(
        assignments=assignments,
        expert_counts=counts,
        load_imbalance=(max(counts) / avg) if avg else 0.0,
        cv=(sd / avg) if avg else 0.0,
        empty_expert_ratio=(empty / len(counts)) if counts else 0.0,
    )


def _weights(num_experts: int, routing: RoutingConfig) -> list[float]:
    if routing.distribution == "uniform":
        return [1.0] * num_experts
    if routing.distribution == "zipf":
        return [1.0 / ((i + 1) ** routing.zipf_alpha) for i in range(num_experts)]
    if routing.distribution == "hotspot":
        hot = max(1, min(routing.hotspot_experts, num_experts))
        cold = max(1, num_experts - hot)
        hot_w = routing.hotspot_fraction / hot
        cold_w = (1.0 - routing.hotspot_fraction) / cold
        return [hot_w if i < hot else cold_w for i in range(num_experts)]
    raise ValueError(f"Unknown routing distribution: {routing.distribution}")


def _weighted_sample_without_replacement(
    rng: random.Random, population: int, k: int, weights: list[float]
) -> list[int]:
    chosen: list[int] = []
    available = list(range(population))
    current_weights = list(weights)
    for _ in range(min(k, population)):
        total = sum(current_weights[i] for i in available)
        pick = rng.random() * total
        acc = 0.0
        selected = available[-1]
        for idx in available:
            acc += current_weights[idx]
            if acc >= pick:
                selected = idx
                break
        chosen.append(selected)
        available.remove(selected)
    return chosen
