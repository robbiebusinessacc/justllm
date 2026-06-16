"""Representative "dynamic context" payloads — the junk that bloats agent calls.

Deterministic (no randomness) so benchmark numbers are reproducible run to run.
"""
from __future__ import annotations

import json
from typing import Dict


def json_api_response(n: int = 60) -> str:
    """A verbose paginated API response — the classic tool-output bloat."""
    records = [
        {
            "id": 100000 + i,
            "uuid": f"3f2504e0-4f89-11d3-9a0c-{i:012d}",
            "first_name": ["Ava", "Ben", "Cara", "Dan", "Eve"][i % 5],
            "last_name": ["Ng", "Ortiz", "Patel", "Quinn", "Rossi"][i % 5],
            "email": f"user{i}@example.com",
            "created_at": "2026-01-15T09:24:01.000Z",
            "updated_at": "2026-06-14T22:10:55.000Z",
            "status": "active" if i % 3 else "suspended",
            "role": "member",
            "metadata": {"plan": "pro", "seats": (i % 7) + 1, "region": "us-east-1"},
            "address": {
                "line1": f"{i} Market St",
                "city": "Boston",
                "state": "MA",
                "zip": "02110",
                "country": "US",
            },
        }
        for i in range(n)
    ]
    return json.dumps(
        {"page": 1, "per_page": n, "total": 4213, "data": records}, indent=2
    )


def server_logs(n: int = 200) -> str:
    """Application logs — mostly repetitive INFO with a few real signals."""
    lines = []
    for i in range(n):
        ts = f"2026-06-15T14:{(i // 60) % 60:02d}:{i % 60:02d}.123Z"
        if i % 47 == 0:
            lines.append(
                f"{ts} ERROR worker-3 db.timeout query took 5123ms pool exhausted"
            )
        elif i % 13 == 0:
            lines.append(
                f"{ts} WARN  api.gateway rate-limit near threshold tenant=acme 92%"
            )
        else:
            lines.append(
                f"{ts} INFO  http.request GET /v1/users 200 12ms trace={i:08d}"
            )
    return "\n".join(lines)


def rag_chunks(n: int = 12) -> str:
    """Retrieved document chunks with the usual boilerplate headers/footers."""
    boiler = (
        "\n\n---\nThis document is confidential and proprietary. "
        "Copyright 2026. All rights reserved. Do not distribute.\n---\n"
    )
    chunks = []
    for i in range(n):
        chunks.append(
            f"[chunk {i + 1}/{n}] Source: handbook_v4.pdf p.{i + 3}\n"
            "The reimbursement policy states that employees may submit expenses "
            "within 30 days. Approvals route to the line manager and then to "
            "finance. Receipts are required for any single item over the stated "
            "threshold. " + boiler
        )
    return "\n".join(chunks)


def code_file() -> str:
    """A source file — comments + imports + boilerplate the model rarely needs."""
    block = '''\
# -*- coding: utf-8 -*-
# Copyright 2026. Licensed under the Apache License, Version 2.0.
import os
import sys
import json
import logging

logger = logging.getLogger(__name__)


def process(records):
    """Process a batch of records and return the valid ones.

    A long docstring that restates the obvious for the benefit of documentation
    generators that nobody reads in practice.
    """
    valid = []
    for r in records:
        if r.get("status") == "active":
            valid.append(r)
    return valid
'''
    return block * 4


FIXTURES = {
    "json_api_response": json_api_response,
    "server_logs": server_logs,
    "rag_chunks": rag_chunks,
    "code_file": code_file,
}


def all_fixtures() -> Dict[str, str]:
    return {name: fn() for name, fn in FIXTURES.items()}
