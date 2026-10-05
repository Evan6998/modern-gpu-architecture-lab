"""Supplied host-only ring model; NOT the kernel's mbarrier implementation.

Ticket parity is a conceptual ownership epoch, not an instruction telling you
which hardware barrier phase to initialize or wait on.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Ticket:
    stage: int
    epoch: int
    parity: int


def ticket(iteration: int, stages: int) -> Ticket:
    if iteration < 0 or stages < 1:
        raise ValueError("iteration >= 0 and stages >= 1 required")
    epoch, stage = divmod(iteration, stages)
    return Ticket(stage, epoch, epoch % 2)
