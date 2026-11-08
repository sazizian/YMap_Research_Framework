"""PostgreSQL execution of Appendix A with explicit choices for missing inputs.

A new measurement is not the historical experiment. Smoke labels deliberately
cannot be used as research labels by the existing dataset validator.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import datetime as dt
from decimal import Decimal
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import string
import subprocess
import time
import uuid

from .common import validate_output_schema_ids
from .profiling import ProfilingError, _digest, build_load_plan
from .synthesis import SQL_TYPES, _literal, _name, model_index, quote, validate_model

TEMPLATES = ("T1", "T2", "T3", "T4", "T5")
TEMPLATE_NAMES = dict(zip(TEMPLATES, ("primary-key lookup", "subclass lookup",
    "one-to-many traversal", "multi-table retrieval", "bounded aggregation")))
VARIANTS = ("reference", "W_lookup", "W_join", "D_skew", "D_10x")
REFERENCE_SETTINGS = {
    "shared_buffers": 64 * 1024**3, "work_mem": 64 * 1024**2,
    "maintenance_work_mem": 2 * 1024**3, "effective_cache_size": 192 * 1024**3,
    "max_parallel_workers_per_gather": 4, "fsync": "on", "synchronous_commit": "on",
    "autovacuum": "off", "jit": "off", "random_page_cost": 1.1,
    "seq_page_cost": 1.0, "default_statistics_target": 100,
    "checkpoint_timeout": 900_000, "max_wal_size": 8 * 1024**3,
}
_SETTING_UNITS = {"B": 1, "kB": 1024, "MB": 1024**2, "GB": 1024**3,
                  "8kB": 8192, "ms": 1, "s": 1000, "min": 60000}
_NUMERIC = {"int", "integer", "decimal", "float", "double"}
_PROTOCOL = "ymap_appendix_a_configured_v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ProfilingError(f"{name} must be an integer >= {minimum}.")
    return value


def _setting_number(setting, unit):
    return float(setting) * _SETTING_UNITS.get(unit, 1)


def _configuration(config):
    if not isinstance(config, dict) or config.get("mode") not in ("reference", "smoke"):
        raise ProfilingError("Set profiling mode to reference or smoke.")
    if config.get("confirm_disposable_server") is not True:
        raise ProfilingError("Set confirm_disposable_server: true only for a PostgreSQL server you authorize for disposable profiling.")
    mode = config["mode"]
    variant = config.get("variant", "reference")
    if variant not in VARIANTS:
        raise ProfilingError(f"Unknown profiling variant: {variant!r}.")
    if type(config.get("seed")) is not int:
        raise ProfilingError("Specify an integer seed; historical seeds are unavailable.")
    for key in ("population_by_system", "workload_by_system"):
        if not isinstance(config.get(key), dict) or not config[key]:
            raise ProfilingError(f"Specify {key} explicitly; historical inputs are unavailable.")
    expected = {"repetitions": 5, "warmup_operations": 1000,
                "insert_operations": 10_000, "query_operations": 10_000}
    defaults = expected if mode == "reference" else dict(
        repetitions=2, warmup_operations=10, insert_operations=20, query_operations=30)
    counts = {}
    for key, value in defaults.items():
        counts[key] = _integer(config.get(key, value), key, 0 if key == "warmup_operations" else 1)
        if mode == "reference" and counts[key] != expected[key]:
            raise ProfilingError(f"{key} must be {expected[key]} in reference mode.")
    warmup_inserts = _integer(config.get("warmup_inserts", counts["warmup_operations"] // 2 if mode == "smoke" else None), "warmup_inserts")
    if warmup_inserts > counts["warmup_operations"]:
        raise ProfilingError("warmup_inserts cannot exceed warmup_operations.")
    if mode == "reference" and not 0 < warmup_inserts < counts["warmup_operations"]:
        raise ProfilingError("Reference warm-up must include both insert and query operations.")
    weights = config.get("template_weights")
    if variant in {"W_lookup", "W_join"}:
        favored = ("T1", "T2") if variant == "W_lookup" else ("T3", "T4")
        message = f"{variant} needs explicit positive percentages for T1–T5 (up to 2 decimal places), summing to 100 with 80 in {favored}."
        if not isinstance(weights, dict) or set(weights) != set(TEMPLATES) or any(
                type(v) not in (int, float) or not 0 < v < 100 for v in weights.values()):
            raise ProfilingError(message)
        precise = {k: Decimal(str(v)) for k, v in weights.items()}
        if (any(v != v.quantize(Decimal("0.01")) for v in precise.values())
                or sum(precise.values()) != 100 or sum(precise[k] for k in favored) != 80):
            raise ProfilingError(message)
    else:
        if weights is not None and weights != dict.fromkeys(TEMPLATES, 20):
            raise ProfilingError("Reference and data variants use exactly 20 percent per template.")
        weights = dict.fromkeys(TEMPLATES, 20)
    reset = config.get("cache_reset_command")
    if reset is not None and (not isinstance(reset, list) or not reset
            or any(not isinstance(v, str) or not v or "\x00" in v for v in reset)
            or not Path(reset[0]).is_absolute()):
        raise ProfilingError("cache_reset_command must be a trusted argv list with an absolute executable path.")
    notes = config.get("environment_notes")
    if reset and config.get("confirm_dedicated_host") is not True:
        raise ProfilingError("A cache reset can affect other databases; set confirm_dedicated_host: true only for an authorized dedicated host.")
    if mode == "reference":
        if not reset:
            raise ProfilingError("Reference mode requires confirm_dedicated_host: true and an authorized cache_reset_command that restarts PostgreSQL and clears the OS page cache.")
        if not isinstance(notes, str) or not notes.strip():
            raise ProfilingError("Record the dedicated server hardware, operating system and storage in environment_notes.")
    dsn_env = config.get("dsn_env", "YMAP_POSTGRES_DSN")
    if not isinstance(dsn_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", dsn_env):
        raise ProfilingError("dsn_env must name an environment variable.")
    statement_timeout = _integer(config.get("statement_timeout_ms", 0 if mode == "reference" else 120_000), "statement_timeout_ms")
    reset_timeout = _integer(config.get("cache_reset_timeout_seconds", 120), "cache_reset_timeout_seconds", 1)
    return dict(mode=mode, variant=variant, seed=config["seed"], dsn_env=dsn_env,
                **counts, warmup_inserts=warmup_inserts, template_weights=weights,
                cache_reset_command=reset, cache_reset_timeout_seconds=reset_timeout,
                statement_timeout_ms=statement_timeout, environment_notes=notes,
                population_by_system=config["population_by_system"],
                workload_by_system=config["workload_by_system"])


def _system_config(model, config):
    validate_model(model)
    name, idx = model["name"], model_index(model)
    population = config["population_by_system"].get(name)
    if not isinstance(population, dict) or set(population) != set(idx["concrete"]):
        raise ProfilingError(f"{name}: population_by_system must list every exact concrete class.")
    for cls, n in population.items():
        _integer(n, f"{name}.{cls} population", 1)
    for root in idx["roots"]:
        counts = [population[c] for c in idx["descendants"][root] if c in population]
        if max(counts) - min(counts) > 1:
            raise ProfilingError(f"{name}: split the population uniformly over concrete classes of {root}.")
    workload = config["workload_by_system"].get(name)
    if not isinstance(workload, dict) or set(workload) != {*TEMPLATES, "insert_classes"}:
        raise ProfilingError(f"{name}: specify T1–T5 and insert_classes; missing retrieval intents are not replaced.")
    if any(not isinstance(workload[k], dict) for k in TEMPLATES):
        raise ProfilingError(f"{name}: each template requires a configuration object.")
    classes = workload["insert_classes"]
    if (not isinstance(classes, list) or not classes or any(not isinstance(c, str) or c not in population for c in classes)
            or len(set(classes)) != len(classes)):
        raise ProfilingError(f"{name}: insert_classes must list unique concrete classes.")
    first, second = workload["T1"].get("class"), workload["T2"].get("class")
    if (set(workload["T1"]) != {"class"} or set(workload["T2"]) != {"class"}
            or not isinstance(first, str) or not isinstance(second, str)
            or first not in population or second not in population or first == second or len(idx["paths"][second]) < 2):
        raise ProfilingError(f"{name}: T1 requires a concrete class and T2 a distinct concrete subclass.")
    associations = {a["name"]: a for a in model.get("associations", [])}

    def hop(association, direction="forward"):
        if not isinstance(association, str) or association not in associations or direction not in ("forward", "reverse"):
            raise ProfilingError(f"{name}: choose a declared association and forward/reverse direction.")
        info = associations[association]
        ends = (info["source"], info["target"])
        return {"name": association, "direction": direction,
                "start": ends[direction == "reverse"], "end": ends[direction == "forward"]}

    if set(workload["T3"]) != {"association"}:
        raise ProfilingError(f"{name}: T3 requires a forward one-to-many association.")
    t3 = hop(workload["T3"]["association"])
    if associations[t3["name"]].get("multiplicity", "one-to-many") != "one-to-many":
        raise ProfilingError(f"{name}: T3 requires a forward one-to-many association.")
    t4 = workload["T4"]
    path, directions = t4.get("associations"), t4.get("directions", ["forward", "forward"])
    if (set(t4) - {"associations", "directions"} or not isinstance(path, list) or len(path) != 2
            or path[0] == path[1] or not isinstance(directions, list) or len(directions) != 2):
        raise ProfilingError(f"{name}: T4 requires two distinct declared associations, with two directions if provided.")
    h1, h2 = (hop(a, d) for a, d in zip(path, directions))
    if h1["end"] != h2["start"]:
        raise ProfilingError(f"{name}: T4 associations must share the intermediate logical endpoint class.")
    t5 = workload["T5"]
    if not {"association", "attribute", "operation"} <= set(t5) or set(t5) - {"association", "attribute", "operation", "direction"}:
        raise ProfilingError(f"{name}: T5 requires association, attribute and operation.")
    h5 = hop(t5["association"], t5.get("direction", "forward"))
    fields = {a["name"]: (owner, a) for owner in idx["paths"][h5["end"]]
              for a in idx["classes"][owner].get("attributes", [])}
    attr, op = t5["attribute"], t5["operation"]
    if (not isinstance(attr, str) or attr not in fields or
            fields[attr][1].get("type", "string").lower() not in _NUMERIC or op not in ("sum", "avg", "min", "max")):
        raise ProfilingError(f"{name}: T5 requires a numeric attribute of {h5['end']} and a sum/avg/min/max operation.")
    owner, field = fields[attr]
    return dict(population=population, insert_classes=classes, T1=first, T2=second, T3=t3, T4=(h1, h2),
                T5={**h5, "column": _name("a_", owner + "_" + attr), "operation": op,
                    "type": field.get("type", "string").lower()})


def _seed(config, *parts):
    return int(_digest([config["seed"], *parts])[:16], 16)


def _new_object(cls, oid, idx, rng):
    values = {}
    for owner in idx["paths"][cls]:
        for attr in idx["classes"][owner].get("attributes", []):
            kind = attr.get("type", "string").lower()
            if attr.get("nullable") and rng.randrange(10) == 0:
                value = None
            elif kind in {"int", "integer"}:
                value = rng.randrange(1, 1_000_001)
            elif kind in {"double", "float"}:
                value = rng.uniform(0.01, 10_000)
            elif kind == "decimal":
                value = Decimal(rng.randrange(100, 100_000_001)).scaleb(-4)
            elif kind in {"bool", "boolean"}:
                value = bool(rng.randrange(2))
            elif kind == "date":
                value = dt.date(2020, 1, 1) + dt.timedelta(days=rng.randrange(2000))
            elif kind == "datetime":
                value = dt.datetime(2020, 1, 1) + dt.timedelta(seconds=rng.randrange(100_000_000))
            else:
                value = "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randrange(8, 33)))
            values[_name("a_", owner + "_" + attr["name"])] = value
    return {"id": oid, "class": cls, "attributes": values}


class _TargetPool:
    """Draw distinct eligible targets in O(log n) per draw; cap both endpoints."""

    def __init__(self, keys, limit, skew):
        self.keys = keys
        self.remaining = [limit] * len(keys)
        self.weights = [max(1, round(10**12 / (i + 1)**1.1)) if skew else 1 for i in range(len(keys))]
        self.tree = [0] * (len(keys) + 1)
        for i, weight in enumerate(self.weights):
            self._update(i, weight)
        self.total, self.active = sum(self.weights), len(keys)

    def _update(self, pos, change):
        pos += 1
        while pos < len(self.tree):
            self.tree[pos] += change
            pos += pos & -pos

    def _index(self, threshold):
        idx, bit = 0, 1 << (len(self.tree).bit_length() - 1)
        while bit:
            j = idx + bit
            if j < len(self.tree) and self.tree[j] <= threshold:
                idx, threshold = j, threshold - self.tree[j]
            bit >>= 1
        return idx

    def sample(self, count, rng):
        positions = []
        for _ in range(count):
            i = self._index(rng.randrange(self.total))
            positions.append(i)
            self.total -= self.weights[i]
            self._update(i, -self.weights[i])
            self.remaining[i] -= 1
        for i in positions:
            if self.remaining[i] > 0:
                self.total += self.weights[i]
                self._update(i, self.weights[i])
            else:
                self.active -= 1
        return [self.keys[i] for i in positions]


def _degree(rng, low, high, skew):
    if not skew:
        return rng.randint(low, high)
    values = range(low, high + 1)
    return rng.choices(values, weights=[(n + 1)**-1.1 for n in values], k=1)[0]


def _links(association, sources, targets, rng, skew):
    sources, targets = list(sources), list(targets)
    rng.shuffle(sources)
    rng.shuffle(targets)
    mult = association.get("multiplicity", "one-to-many")
    if mult == "many-to-many":
        pool = _TargetPool(targets, min(10, len(sources)), skew)
        return [[source, target] for source in sources
                for target in pool.sample(_degree(rng, 0, min(10, pool.active), skew), rng)]
    cap = 1 if mult == "one-to-one" else 10
    if association.get("nullable", False):
        possible = min(len(targets), cap * len(sources))
        # Sample each optional target participation, conditioned on feasible capacity.
        total = min(possible, sum(_degree(rng, 0, 1, skew) for _ in targets))
        targets = _TargetPool(targets, 1, skew).sample(total, rng)
    if len(targets) > cap * len(sources):
        raise ProfilingError(f"{association['name']}: initial population exceeds source capacity {cap}; supply feasible counts.")
    slots, remain = [], len(targets)
    if skew:
        # Use ranked references while still meeting mandatory target participation.
        pool = _TargetPool(sources, cap, True)
        degrees = [0] * len(sources)
        for i in range(len(sources)):
            degrees[i] = _degree(rng, 0, cap, True)
        # Seed the capped source slots with Zipf degrees, then use weighted
        # sampling for residual references that must exist for the fixed total.
        ranked = [s for s, degree in zip(sources, degrees) for _ in range(degree)]
        rng.shuffle(ranked)
        slots = ranked[:remain]
        initial_degrees = Counter(slots)
        for i, source in enumerate(sources):
            pool.remaining[i] -= initial_degrees[source]
            if not pool.remaining[i]:
                pool._update(i, -pool.weights[i])
                pool.total -= pool.weights[i]
                pool.active -= 1
        slots.extend(pool.sample(1, rng)[0] for _ in range(remain - len(slots)))
    else:
        for i, source in enumerate(sources):
            degree = _degree(rng, max(0, remain - cap * (len(sources) - i - 1)), min(cap, remain), False)
            slots.extend([source] * degree)
            remain -= degree
    rng.shuffle(slots)
    return [[source, target] for source, target in zip(slots, targets)]


def _fixture(model, population, seed, variant):
    idx = model_index(model)
    rng = random.Random(seed)
    objects, class_ids = [], defaultdict(list)
    scale = 10 if variant == "D_10x" else 1
    for cls in idx["concrete"]:
        for _ in range(population[cls]):
            oid = len(objects) + 1
            objects.append(_new_object(cls, oid, idx, rng))
            for ancestor in idx["paths"][cls]:
                class_ids[ancestor].append(oid)
    links = {}
    for assoc in sorted(model.get("associations", []), key=lambda a: a["name"]):
        links[assoc["name"]] = _links(assoc, class_ids[assoc["source"]], class_ids[assoc["target"]],
                                      random.Random(int(_digest([seed, assoc["name"]])[:16], 16)),
                                      variant == "D_skew")
    if scale == 10:
        # Clone the logical topology to preserve exactly 10x link-table row
        # counts, randomize per-class ID assignments, and draw new attributes.
        base_size = len(objects)
        base_classes = [o["class"] for o in objects]
        permutations = {}
        for cls in idx["concrete"]:
            ids = [i + copy * base_size for copy in range(scale)
                   for i, c in enumerate(base_classes, 1) if c == cls]
            shuffled = list(ids)
            rng.shuffle(shuffled)
            permutations.update(zip(ids, shuffled))
        objects = [_new_object(cls, oid, idx, rng)
                   for oid, cls in enumerate(base_classes * scale, 1)]
        links = {name: [[permutations[s + k * base_size], permutations[t + k * base_size]]
                        for k in range(scale) for s, t in pairs] for name, pairs in links.items()}
    return {"seed": seed, "objects": objects, "links": links,
            "rows_per_class": {k: scale * v for k, v in population.items()},
            "realization_id": _digest({"model": model, "objects": objects, "links": links})}


class _InsertSlots:
    def __init__(self, cap, ids, used):
        self.cap = cap
        self.available, self.positions, self.used = [], {}, Counter(used)
        for key in ids:
            self.add(key)

    def add(self, key):
        if self.used[key] < self.cap:
            self.positions[key] = len(self.available)
            self.available.append(key)

    def draw(self, rng, association):
        if not self.available:
            raise ProfilingError(f"{association}: a requested insert has no source capacity; revise population or insert_classes.")
        key = rng.choice(self.available)
        self.used[key] += 1
        if self.used[key] == self.cap:
            pos = self.positions.pop(key)
            last = self.available.pop()
            if pos < len(self.available):
                self.available[pos] = last
                self.positions[last] = pos
        return key


def _insert_objects(model, fixture, classes, count, seed):
    idx = model_index(model)
    rng = random.Random(seed)
    class_ids = defaultdict(list)
    for obj in fixture["objects"]:
        for cls in idx["paths"][obj["class"]]:
            class_ids[cls].append(obj["id"])
    required = [a for a in sorted(model.get("associations", []), key=lambda a: a["name"])
                if a.get("multiplicity") != "many-to-many" and not a.get("nullable", False)]
    slots = {a["name"]: _InsertSlots(1 if a.get("multiplicity") == "one-to-one" else 10,
                                    class_ids[a["source"]], [s for s, _ in fixture["links"][a["name"]]])
             for a in required}
    by_id = {o["id"]: o for o in fixture["objects"]}
    records = []
    for oid in range(len(by_id) + 1, len(by_id) + count + 1):
        cls = rng.choice(classes)
        obj = _new_object(cls, oid, idx, rng)
        path = idx["paths"][cls]
        incoming = {}
        for assoc in required:
            if assoc["target"] in path:
                source = slots[assoc["name"]].draw(rng, assoc["name"])
                incoming[assoc["name"]] = (source, by_id[source]["class"])
        for assoc in required:
            if assoc["source"] in path:
                slots[assoc["name"]].add(oid)
        by_id[oid] = obj
        records.append((obj, incoming))
    return records


def _query_bindings(workload, fixture, idx, count, weights, phase_seed):
    exact = defaultdict(list)
    poly = defaultdict(list)
    for obj in fixture["objects"]:
        exact[obj["class"]].append(obj["id"])
        for c in idx["paths"][obj["class"]]:
            poly[c].append(obj["id"])
    keys = {"T1": exact[workload["T1"]], "T2": exact[workload["T2"]],
            "T3": poly[workload["T3"]["start"]], "T4": poly[workload["T4"][0]["start"]],
            "T5": poly[workload["T5"]["start"]]}
    numerators = {key: count * Decimal(str(weights[key])) for key in TEMPLATES}
    slots = {key: int(numerators[key] // 100) for key in TEMPLATES}
    residuals = sorted(TEMPLATES, key=lambda key: (-(numerators[key] % 100), key))
    for key in residuals[:count - sum(slots.values())]:
        slots[key] += 1
    bindings = []
    for key in TEMPLATES:
        rng = random.Random(int(_digest([phase_seed, key])[:16], 16))
        bindings.extend((key, rng.choice(keys[key])) for _ in range(slots[key]))
    random.Random(phase_seed).shuffle(bindings)
    return bindings


def _fields(idx, cls, exact=False):
    owners = set(idx["paths"][cls])
    if not exact:
        owners.update(idx["descendants"][cls])
    return [(_name("a_", owner + "_" + attr["name"]), SQL_TYPES[attr.get("type", "string").lower()], owner)
            for owner in sorted(owners)
            for attr in sorted(idx["classes"][owner].get("attributes", []), key=lambda a: a["name"])]


def _relation(candidate, cls, idx, exact=False):
    fields = _fields(idx, cls, exact)
    branches = []
    concretes = [cls] if exact else [c for c in idx["descendants"][cls] if c in idx["concrete"]]
    if candidate["mapping"][concretes[0]]["strategy"] == "single_table":
        table = quote(candidate["mapping"][concretes[0]]["table"])
        condition = ", ".join(_literal(c) for c in concretes)
        return "SELECT " + ", ".join(['p."id"', 'p."dtype"', *["p." + quote(col) for col, _, _ in fields]]) + f' FROM {table} p WHERE p."dtype" IN ({condition})'
    for concrete in concretes:
        path = idx["paths"][concrete]
        m = candidate["mapping"][concrete]
        if m["strategy"] == "joined":
            aliases = {owner: "p" + str(i) for i, owner in enumerate(path)}
            root = aliases[path[0]]
            select = [root + '."id" AS "id"', root + '."dtype" AS "dtype"']
            source = quote(candidate["mapping"][path[0]]["table"]) + " " + root
            for owner in path[1:]:
                alias = aliases[owner]
                source += " JOIN " + quote(candidate["mapping"][owner]["table"]) + f' {alias} ON {alias}."id"={root}."id"'
            condition = f' WHERE {root}."dtype"=' + _literal(concrete)
        else:
            aliases = dict.fromkeys(path, "p")
            select = ['p."id" AS "id"', f'{_literal(concrete)}::text AS "dtype"']
            source = quote(m["table"]) + " p"
            condition = ""
        for col, kind, owner in fields:
            expr = aliases[owner] + "." + quote(col) if owner in aliases else "NULL::" + kind
            select.append(expr + " AS " + quote(col))
        branches.append("SELECT " + ", ".join(select) + " FROM " + source + condition)
    return " UNION ALL ".join(branches)


def _link(candidate, hop):
    am = candidate["association_mapping"][hop["name"]]
    cols = [quote(am[k][0]) for k in ("source_columns", "target_columns")]
    left, right = cols if hop["direction"] == "forward" else cols[::-1]
    return f"SELECT {left} AS lhs, {right} AS rhs FROM {quote(am['table'])} WHERE {left} IS NOT NULL AND {right} IS NOT NULL"


def _query_sql(candidate, workload):
    idx = model_index(candidate["source_model"])

    def relation(cls, exact=False):
        return "(" + _relation(candidate, cls, idx, exact=exact) + ")"

    def link(hop):
        return "(" + _link(candidate, hop) + ")"

    result = {t: f'SELECT q.* FROM {relation(workload[t], exact=True)} q WHERE q."id"=%s'
              for t in ("T1", "T2")}
    t3 = workload["T3"]
    result["T3"] = f'SELECT b.* FROM {link(t3)} x JOIN {relation(t3["end"])} b ON b."id"=x.rhs WHERE x.lhs=%s ORDER BY b."id"'
    h1, h2 = workload["T4"]
    result["T4"] = (
        f'SELECT a.*, b.*, c.* FROM {relation(h1["start"])} a '
        f'LEFT JOIN {link(h1)} x ON x.lhs=a."id" LEFT JOIN {relation(h1["end"])} b ON b."id"=x.rhs '
        f'LEFT JOIN {link(h2)} y ON y.lhs=b."id" LEFT JOIN {relation(h2["end"])} c ON c."id"=y.rhs '
        'WHERE a."id"=%s ORDER BY b."id" NULLS LAST, c."id" NULLS LAST')
    t5 = workload["T5"]
    result["T5"] = (f'SELECT {t5["operation"]}(b.{quote(t5["column"])}) FROM {link(t5)} x '
                    f'JOIN {relation(t5["end"])} b ON b."id"=x.rhs WHERE x.lhs=%s')
    return result


def _oracle(model, fixture, inserts, workload, bindings):
    idx = model_index(model)
    objects = {obj["id"]: obj for obj in fixture["objects"]}
    links = {k: list(pairs) for k, pairs in fixture["links"].items()}
    for obj, incoming in inserts:
        objects[obj["id"]] = obj
        for association, (source, _) in incoming.items():
            links[association].append([source, obj["id"]])

    def relation(hop):
        result = defaultdict(list)
        for a, b in links[hop["name"]]:
            if hop["direction"] == "reverse":
                a, b = b, a
            result[a].append(b)
        return {key: sorted(values) for key, values in result.items()}

    h1, h2 = workload["T4"]
    adj3, adj41, adj42, adj5 = [relation(h) for h in (workload["T3"], h1, h2, workload["T5"])]
    if not adj3 or not any(b in adj42 for row in adj41.values() for b in row) or not adj5:
        raise ProfilingError(f"{model['name']}: fixture cannot realize all five intents with the declared associations.")
    if any(len(row) > 10 for row in adj3.values()):
        raise ProfilingError(f"{model['name']}: T3 would return more than 10 rows.")
    if any(sum(max(1, len(adj42.get(b, []))) for b in row) > 100 for row in adj41.values()):
        raise ProfilingError(f"{model['name']}: T4 would return more than 100 rows.")
    if any(len(row) > 10 for row in adj5.values()):
        raise ProfilingError(f"{model['name']}: T5 would aggregate over more than 10 rows.")
    projection = {(cls, exact): [f[0] for f in _fields(idx, cls, exact)]
                  for cls, exact in {(workload["T1"], True), (workload["T2"], True),
                                     (workload["T3"]["end"], False), (h1["start"], False),
                                     (h1["end"], False), (h2["end"], False), (workload["T5"]["end"], False)}}

    def row(oid, cls, exact=False):
        fields = projection[cls, exact]
        if oid is None:
            return (None,) * (2 + len(fields))
        obj = objects[oid]
        return (oid, obj["class"], *(obj["attributes"].get(col) for col in fields))

    answers = {}
    for template, key in sorted(set(bindings)):
        if template in ("T1", "T2"):
            result = [row(key, workload[template], exact=True)]
        elif template == "T3":
            result = [row(b, workload["T3"]["end"]) for b in adj3.get(key, [])]
        elif template == "T4":
            result = [row(key, h1["start"]) + row(b, h1["end"]) + row(c, h2["end"])
                      for b in adj41.get(key, [None]) for c in adj42.get(b, [None])]
        else:
            t5 = workload["T5"]
            values = [objects[b]["attributes"][t5["column"]] for b in adj5.get(key, [])]
            values = [v for v in values if v is not None]
            value = None
            if values:
                if t5["operation"] == "avg":
                    value = sum(values) / len(values) if isinstance(values[0], float) else Decimal(sum(values)) / len(values)
                else:
                    value = {"sum": sum, "min": min, "max": max}[t5["operation"]](values)
            result = [(value,)]
        answers[template, key] = result
    return answers


def _realization(model, workload, config, repeat):
    name = model["name"]
    idx = model_index(model)
    seeds = {phase: _seed(config, name, repeat, phase) for phase in ("initial", "inserts", "warmup_queries", "timed_queries")}
    fixture = _fixture(model, workload["population"], seeds["initial"], config["variant"])
    n_warm = config["warmup_inserts"]
    inserts = _insert_objects(model, fixture, workload["insert_classes"], n_warm + config["insert_operations"], seeds["inserts"])
    warm = _query_bindings(workload, fixture, idx, config["warmup_operations"] - n_warm, config["template_weights"], seeds["warmup_queries"])
    timed = _query_bindings(workload, fixture, idx, config["query_operations"], config["template_weights"], seeds["timed_queries"])
    if {key for key, _ in timed} != set(TEMPLATES):
        raise ProfilingError("Increase query_operations so the timed phase includes all five templates at the selected weights.")
    oracle_warm = _oracle(model, fixture, inserts[:n_warm], workload, warm) if warm else {}
    oracle_timed = _oracle(model, fixture, inserts, workload, timed)
    return dict(fixture=fixture, inserts=inserts, warm=warm, timed=timed,
                oracle_warm=oracle_warm, oracle_timed=oracle_timed, seeds=seeds,
                realization_id=_digest([fixture["realization_id"], inserts, warm, timed]))


def _insert_rows(candidate, idx, record):
    obj, incoming = record
    cls, oid = obj["class"], obj["id"]
    m = candidate["mapping"][cls]
    if m["strategy"] == "single_table":
        rows = {m["table"]: {"id": oid, "dtype": cls, **obj["attributes"]}}
    elif m["strategy"] == "joined":
        rows = {}
        for owner in idx["paths"][cls]:
            attrs = {_name("a_", owner + "_" + a["name"]): obj["attributes"][_name("a_", owner + "_" + a["name"])]
                     for a in idx["classes"][owner].get("attributes", [])}
            rows[candidate["mapping"][owner]["table"]] = {"id": oid, "dtype": cls, **attrs}
    else:
        rows = {m["table"]: {"id": oid, **obj["attributes"]}}
    for association, (source_id, source_class) in incoming.items():
        am = candidate["association_mapping"][association]
        source = dict(zip(am["source_columns"], (source_id, source_class)))
        if am["strategy"] == "direct_fk":
            rows[am["table"]].update(source)
        else:
            target = dict(zip(am["target_columns"], (oid, cls)))
            rows[am["table"]] = {**source, **target}
    return rows


def _same_rows(observed, expected):
    if len(observed) != len(expected):
        return False
    for actual, wanted in zip(observed, expected):
        if len(actual) != len(wanted):
            return False
        for a, b in zip(actual, wanted):
            if a == b:
                continue
            if a is None or b is None or not isinstance(a, (float, Decimal)) or not isinstance(b, (float, Decimal)):
                return False
            if not math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-14):
                return False
    return True


def _schedule(size, repetitions, seed):
    rng = random.Random(seed)
    last = None
    for repeat in range(repetitions):
        order = list(range(size))
        rng.shuffle(order)
        if size > 1 and order[0] == last:
            order.append(order.pop(0))
        for index in order:
            yield repeat, index
        last = order[-1]


def _json(path, payload):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _event(handle, **payload):
    handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    handle.flush()


def _environment(conn, config, driver_version):
    watched = sorted({*REFERENCE_SETTINGS, "plan_cache_mode", "TimeZone", "server_encoding",
                      "server_version_num", "block_size", "default_transaction_isolation",
                      "effective_io_concurrency", "shared_preload_libraries", "wal_level"})
    with conn.cursor() as cur:
        cur.execute("SELECT version(), inet_server_addr() IS NULL, pg_postmaster_start_time(), current_database()")
        version, socket, started, database = cur.fetchone()
        cur.execute("SELECT name, setting, unit FROM pg_settings WHERE name=ANY(%s)", (watched,))
        current = {name: {"setting": value, "unit": unit} for name, value, unit in cur.fetchall()}
    if "postgresql" not in version.lower() or any(text in version.lower() for text in ("pglite", "wasm")):
        raise ProfilingError("This protocol requires a native PostgreSQL server.")
    if config["mode"] == "reference":
        if not socket:
            raise ProfilingError("Reference profiling requires a same-host PostgreSQL Unix socket connection.")
        if current["server_version_num"]["setting"] != "140010":
            raise ProfilingError("Reference profiling requires PostgreSQL 14.10. Use smoke mode to test other versions.")
        differences = []
        for key, expected in REFERENCE_SETTINGS.items():
            actual = current[key]["setting"]
            converted = actual if isinstance(expected, str) else _setting_number(actual, current[key]["unit"])
            if expected != converted:
                differences.append(f"{key}={actual} {current[key]['unit'] or ''}".strip())
        if differences:
            raise ProfilingError("PostgreSQL does not match Appendix A settings: " + ", ".join(differences))
    return dict(version=version, unix_socket=socket, settings=current,
                client={"driver": "psycopg", "driver_version": driver_version,
                        "platform": platform.platform(), "python": platform.python_version(),
                        "machine": platform.machine(), "visible_cpus": os.cpu_count()},
                environment_notes=config["environment_notes"]), started, database


def _admin_connect(psycopg, dsn):
    try:
        return psycopg.connect(dsn, autocommit=True, connect_timeout=10)
    except Exception:
        raise ProfilingError("Could not open the PostgreSQL maintenance connection. Check the DSN and authorization.") from None


def _cache_reset(config, lifecycle):
    command = config["cache_reset_command"]
    if command is None:
        return
    env = {key: value for key, value in os.environ.items()
           if key != config["dsn_env"] and key not in ("PGPASSWORD", "PGPASSFILE")}
    try:
        completed = subprocess.run(command, shell=False, env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   check=False, timeout=config["cache_reset_timeout_seconds"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProfilingError(f"Cache reset command could not complete: {type(exc).__name__}.") from None
    if completed.returncode != 0:
        raise ProfilingError(f"Cache reset command returned exit code {completed.returncode}.")
    lifecycle["cache_reset_exit_code"] = completed.returncode


def _fk_indexes(candidate):
    for table in candidate["tables"]:
        columns = sorted({col for fk in table["foreign_keys"] for col in fk["columns"]})
        for col in columns:
            name = "ymap_fk_" + _digest([table["name"], col])[:20]
            yield f"CREATE INDEX {quote(name)} ON {quote(table['name'])} USING btree ({quote(col)})"


def _run_repetition(conn, candidate, sqls, realization, config, stage):
    fixture = realization["fixture"]
    plan = build_load_plan(candidate, fixture)
    idx = model_index(candidate["source_model"])
    with conn.cursor() as cur:
        stage("materialize_schema")
        cur.execute("SET search_path TO public, pg_catalog", prepare=False)
        cur.execute(f"SET statement_timeout TO {config['statement_timeout_ms']}", prepare=False)
        cur.execute(candidate["ddl"], prepare=False)
        for statement in _fk_indexes(candidate):
            cur.execute(statement, prepare=False)
        # The generated canonical DDL's foreign keys and applicable
        # participation checks defer until commit, including cyclic schemas.
        stage("initial_copy")
        with conn.transaction():
            for name, table in plan.items():
                cols = ", ".join(map(quote, table["columns"]))
                with cur.copy(f"COPY {quote(name)} ({cols}) FROM STDIN") as copy:
                    for row in table["rows"]:
                        copy.write_row(row)
        cur.execute("VACUUM ANALYZE", prepare=False)
        cur.execute("SELECT pg_database_size(current_database())", prepare=False)
        storage = int(cur.fetchone()[0])

        def insert(record, timed):
            rows = _insert_rows(candidate, idx, record)
            stmts = [(plan[name]["sql"], tuple(values.get(col) for col in plan[name]["columns"]))
                     for name, values in rows.items()]
            start = time.perf_counter_ns()
            with conn.transaction():
                for sql, params in stmts:
                    cur.execute(sql, params, prepare=True)
            return time.perf_counter_ns() - start if timed else None

        def query(bindings, oracle, timed):
            durations = []
            for key, oid in bindings:
                start = time.perf_counter_ns()
                cur.execute(sqls[key], (oid,), prepare=True)
                observed = cur.fetchall()
                elapsed = time.perf_counter_ns() - start
                if not _same_rows(observed, oracle[key, oid]):
                    raise ProfilingError(f"{candidate['schema_id']}: query {key} returned different logical data for key {oid}; repetition rejected.")
                if timed:
                    durations.append(elapsed)
            return durations

        nw = config["warmup_inserts"]
        stage("warm_up")
        for record in realization["inserts"][:nw]:
            insert(record, False)
        query(realization["warm"], realization["oracle_warm"], False)
        stage("timed_inserts")
        inserts = [insert(record, True) for record in realization["inserts"][nw:]]
        stage("timed_queries")
        queries = query(realization["timed"], realization["oracle_timed"], True)
        cur.execute("SELECT statement FROM pg_prepared_statements", prepare=False)
        prepared = [row[0] for row in cur.fetchall()]
        expected_prepared = {sqls[k].replace("%s", "$1") for k, _ in realization["timed"]}
        if not expected_prepared <= set(prepared):
            raise ProfilingError("Server-side prepared query statements were not observable.")
    return dict(insert_ms=statistics.mean(inserts) / 1e6, query_ms=statistics.mean(queries) / 1e6,
                storage_bytes=storage, storage_mb=storage / 1e6,
                insert_times_ns=inserts, query_times_ns=queries, prepared_statement_count=len(prepared),
                logical_class_counts=fixture["rows_per_class"],
                initial_table_rows={name: len(table["rows"]) for name, table in plan.items()})


def _fresh_database(psycopg, dsn, candidate, sqls, realization, config, baseline_environment, events, repeat):
    db = "ymap_paper_" + uuid.uuid4().hex
    quoted = quote(db)
    lifecycle = {"schema_id": candidate["schema_id"], "repeat": repeat, "database": db}
    _event(events, event="database_create_attempt", **lifecycle)
    oid = None
    create_returned = False
    phase = "create_database"

    def stage(next_phase):
        nonlocal phase
        phase = next_phase
        _event(events, event="stage", phase=phase, **lifecycle)

    try:
        with _admin_connect(psycopg, dsn) as admin:
            with admin.cursor() as cur:
                cur.execute("SELECT pg_postmaster_start_time()")
                pre_reset = cur.fetchone()[0]
                cur.execute(f"CREATE DATABASE {quoted} TEMPLATE template0")
                create_returned = True
                cur.execute("SELECT oid FROM pg_database WHERE datname=%s", (db,))
                oid = int(cur.fetchone()[0])
        _event(events, event="database_created", database_oid=oid, **lifecycle)
        stage("reset_cache")
        _cache_reset(config, lifecycle)
        stage("connect_to_created_database")
        with psycopg.connect(dsn, dbname=db, autocommit=True, connect_timeout=10) as conn:
            env, started, current_name = _environment(conn, config, psycopg.__version__)
            if current_name != db:
                raise ProfilingError("The profiling connection did not select the newly created database.")
            with conn.cursor() as cur:
                cur.execute("SELECT oid FROM pg_database WHERE datname=current_database()")
                if int(cur.fetchone()[0]) != oid:
                    raise ProfilingError("The freshly created database identity changed.")
            if env != baseline_environment:
                raise ProfilingError("PostgreSQL environment changed during profiling.")
            if config["cache_reset_command"] and started == pre_reset:
                raise ProfilingError("Cache reset did not produce a PostgreSQL restart.")
            _event(events, event="database_ready", postmaster_started=started,
                   cache_reset_attested=config["cache_reset_command"] is not None, **lifecycle)
            return _run_repetition(conn, candidate, sqls, realization, config, stage)
    except BaseException as exc:
        _event(events, event="repetition_error", phase=phase, error_type=type(exc).__name__,
               sqlstate=getattr(exc, "sqlstate", None),
               detail=str(exc) if isinstance(exc, ProfilingError) else None, **lifecycle)
        raise
    finally:
        if oid is not None:
            try:
                with _admin_connect(psycopg, dsn) as admin:
                    with admin.cursor() as cur:
                        cur.execute("SELECT oid FROM pg_database WHERE datname=%s", (db,))
                        row = cur.fetchone()
                        if row is None or int(row[0]) != oid:
                            raise ProfilingError("Database identity changed; will not drop it.")
                        cur.execute(f"DROP DATABASE {quoted}")
                _event(events, event="database_dropped", **lifecycle)
            except Exception as exc:
                _event(events, event="database_cleanup_failed", error_type=type(exc).__name__, **lifecycle)
                raise ProfilingError(f"Cleanup failed for the run-created database {db}; consult the lifecycle log.") from None
        elif create_returned:
            _event(events, event="database_not_dropped_identity_unverified", **lifecycle)


def profile_paper(candidates, config, out_dir) -> Path:
    """Run a configured replication protocol; return the measured CSV path.

    This requires operator-selected populations and equivalent logical query
    instantiations. Reference mode enforces known server settings and counts,
    including a user-authorized restart/cache hook on a dedicated server.
    No run can establish the undocumented historical inputs or exact hardware
    equivalence. A failed run leaves evidence but publishes no label CSV.
    """
    config = _configuration(config)
    from .paper_schema import paper_executable_candidate
    candidates = [paper_executable_candidate(record) for record in candidates]
    ids = validate_output_schema_ids(candidates)
    if config["mode"] == "reference" and len(ids) < 2:
        raise ProfilingError("Reference profiling needs two or more candidates to separate each candidate's repetitions.")
    reset = config["cache_reset_command"]
    reset_hash = None
    if reset:
        executable = Path(reset[0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ProfilingError("The cache reset executable is missing or not executable.")
        digest = hashlib.sha256()
        with executable.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        reset_hash = digest.hexdigest()
    dsn = os.environ.get(config["dsn_env"])
    if not dsn:
        raise ProfilingError(f"Set the PostgreSQL connection variable {config['dsn_env']} before profiling.")
    try:
        import psycopg
    except ImportError as exc:
        raise ProfilingError("Install the PostgreSQL extra with pip install '.[postgres]' before profiling.") from exc

    # Validate each mapping and required logical intent before writing to
    # PostgreSQL; after that, retain only the current repetition's fixtures.
    systems, workloads, query_plans, initial = {}, {}, {}, {}
    for candidate in candidates:
        model, name = candidate["source_model"], candidate["object_model"]
        if name in systems and systems[name] != model:
            raise ProfilingError(f"{name}: candidates contain conflicting source models under one system name.")
        if name not in systems:
            systems[name], workloads[name] = model, _system_config(model, config)
            initial[name] = _realization(model, workloads[name], config, 0)
        query_plans[candidate["schema_id"]] = _query_sql(candidate, workloads[name])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {name: out / name for name in ("measurements.csv", "measurements.csv.partial", "workload.json",
                                         "environment.json", "raw_measurements.jsonl.gz", "events.jsonl")}
    if any(path.exists() for path in files.values()):
        raise ProfilingError("Profiling output already exists. Choose a fresh output directory.")
    with _admin_connect(psycopg, dsn) as admin:
        env, _, _ = _environment(admin, config, psycopg.__version__)
    eid = "environment_" + _digest(env)[:16]
    order_seed = _seed(config, "candidate_order")
    schedule = list(_schedule(len(ids), config["repetitions"], order_seed))
    protocol = {key: config[key] for key in ("mode", "variant", "seed", "repetitions", "warmup_operations",
                                             "warmup_inserts", "insert_operations", "query_operations",
                                             "template_weights", "statement_timeout_ms")}
    protocol.update(
        protocol=_PROTOCOL, historical_reproduction=False, template_names=TEMPLATE_NAMES,
        population_by_system={k: workloads[k]["population"] for k in systems},
        workload_by_system={k: config["workload_by_system"][k] for k in systems},
        source_model_hashes={k: _digest(model) for k, model in systems.items()},
        seed_derivation="SHA-256 of JSON arrays with master seed, system, repetition and phase; independent per-template keys",
        cache_reset={"hook_argv_sha256": _digest(reset) if reset else None,
                     "hook_executable_sha256": reset_hash,
                     "server_restart_checked": bool(reset), "os_cache_clear_operator_attested": bool(reset)},
        population_policy={"unbounded_multiplicity_cap": 10,
                           "association_degrees": "sample within bounds conditioned by feasible opposite-end capacity and required participation",
                           "attributes": "uniform within code-defined supports; nullable values independently null with probability 0.1",
                           "skew": "bounded Zipf exponent 1.1 and ranked key references under capacity/participation constraints",
                           "tenfold": "ten copies of initial logical link topology with randomized key assignment and newly sampled attributes; preserves all initial table row multipliers"},
        insert_policy="uniform over configured concrete classes; choose existing eligible required sources uniformly; no optional links",
        warmup_policy="commit warm-up inserts, then execute warm-up queries; retain warm-up writes",
        query_policy="sample uniform keys with replacement from preloaded objects; exact weighted template counts by largest remainder then shuffle; compare full results with independent logical oracle",
        preparation_policy="psycopg prepare=True for insert/query; first preparation included when not encountered in warm-up; database plan_cache_mode recorded",
        measurement_boundaries={"storage": "pg_database_size after COPY + VACUUM ANALYZE, before warm-up; decimal MB",
                                "insert": "one transaction per logical object including COMMIT; physical row construction excluded",
                                "query": "one autocommit statement per query after warm-up and timed inserts, with fetch; no explicit statistics refresh",
                                "aggregation": "means within repetition, medians across repetitions"},
        candidate_order_seed=order_seed,
        candidate_order=[{"repeat": repeat, "schema_id": ids[i]} for repeat, i in schedule],
    )
    wid = "workload_" + _digest({k: v for k, v in protocol.items() if k != "candidate_order"})[:16]
    source = "postgresql_measured" if config["mode"] == "reference" else "postgresql_smoke"
    _json(files["workload.json"], {"workload_id": wid, **protocol, "label_source": source})
    _json(files["environment.json"], {"environment_id": eid, **env})
    summaries = defaultdict(list)
    fields = ["schema_id", "object_model", "insert_ms", "query_ms", "storage_mb",
              "label_source", "workload_id", "environment_id", "seed", "realization_id",
              "repetitions", "query_count", "insert_count", "variant", "profile_protocol"]
    with files["events.jsonl"].open("x", encoding="utf-8") as events, \
            gzip.open(files["raw_measurements.jsonl.gz"], "xt", encoding="utf-8") as raw:
        live_repeat, realized = 0, initial
        for repeat, index in schedule:
            if repeat != live_repeat:
                live_repeat, realized = repeat, {}
                initial.clear()
            candidate = candidates[index]
            sid, name = candidate["schema_id"], candidate["object_model"]
            if name not in realized:
                realized[name] = _realization(systems[name], workloads[name], config, repeat)
            data = realized[name]
            base = dict(schema_id=sid, object_model=name, label_source=source, workload_id=wid,
                        environment_id=eid, seed=config["seed"], realization_id=data["realization_id"],
                        query_count=config["query_operations"], insert_count=config["insert_operations"],
                        variant=config["variant"], profile_protocol=_PROTOCOL)
            stats = _fresh_database(psycopg, dsn, candidate, query_plans[sid], data, config, env, events, repeat)
            _event(raw, **base, repeat=repeat, **stats, seeds=data["seeds"],
                   initial_fixture_id=data["fixture"]["realization_id"],
                   warmup_query_counts=Counter(key for key, _ in data["warm"]),
                   timed_query_counts=Counter(key for key, _ in data["timed"]),
                   timed_query_bindings=data["timed"])
            summaries[sid].append({**base, **{k: stats[k] for k in ("insert_ms", "query_ms", "storage_mb")}})
    with files["measurements.csv.partial"].open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sid in ids:
            records = summaries[sid]
            writer.writerow({**records[0], "repetitions": len(records),
                             "realization_id": _digest([r["realization_id"] for r in records]),
                             **{key: statistics.median(r[key] for r in records)
                                for key in ("insert_ms", "query_ms", "storage_mb")}})
    files["measurements.csv.partial"].replace(files["measurements.csv"])
    return files["measurements.csv"]
