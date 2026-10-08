"""
Sponsor exposure analysis pipeline.

Stages (each reads the previous stage's artifacts from the run directory):

    detect     video   -> rectified advertising bands + detections.jsonl   [done]
    dedup      bands   -> unique creatives                                 [todo]
    identify   creatives -> brands (LLM)                                   [todo]
    review     creatives -> human corrections                              [todo]
    aggregate  all     -> per-brand statistics                             [todo]
    report     stats   -> HTML report                                      [todo]

The split is deliberate: each stage is a pure function over a run directory, so
it is callable from the CLI today and from an API tomorrow without a rewrite.
"""

from .config import RunConfig
from .store import RunStore

__all__ = ["RunConfig", "RunStore"]
