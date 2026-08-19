"""Wire vocabulary between router and workers, and between workers.

Kept as plain strings in a JSON `meta` dict: easy to log, easy to eyeball in a
packet dump, and the KV payload rides alongside as an opaque blob.
"""

from __future__ import annotations

# router -> worker
HELLO = "hello"
SUBMIT = "submit"
CANCEL = "cancel"
SET_PEERS = "set_peers"
SET_LINK = "set_link"
PUSH_PREFIX = "push_prefix"     # "send these cached blocks to peer X"
STATE_REQ = "state_req"
SHUTDOWN = "shutdown"
RESET = "reset"

# worker -> router
HELLO_ACK = "hello_ack"
ADMITTED = "admitted"
REJECTED = "rejected"
TOKEN = "token"
FINISHED = "finished"
PREEMPTED = "preempted"
CACHE_ADD = "cache_add"         # new prefix hashes now resident here
CACHE_DROP = "cache_drop"       # hashes evicted
STATE = "state"
PUSH_DONE = "push_done"
ERROR = "error"

# worker -> worker (data plane)
KV_PUSH = "kv_push"
KV_ACK = "kv_ack"


class SeqState:
    WAITING = "waiting"      # queued, prompt KV not yet computed
    PREFILL = "prefill"      # partially prefilled (chunked)
    RUNNING = "running"      # generating
    DONE = "done"
