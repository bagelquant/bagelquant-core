"""Verified serialized metadata keeps public trees detached and typed."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import sqlite3
from struct import pack
from threading import Barrier

import pytest

from bagelquant_core import NodeMaterialization
from test_indexed_metadata import completed


def _tree(value):
    if isinstance(value, dict):
        return ('dict', tuple((_tree(key), _tree(item)) for key, item in value.items()))
    if isinstance(value, list):
        return ('list', tuple(_tree(item) for item in value))
    if isinstance(value, float):
        return ('float', pack('!d', value))
    return type(value).__name__, value


@pytest.fixture
def serialized(tmp_path):
    _, store, root, receipt = completed(tmp_path)
    original = store.read(receipt['results']['main'][root])
    assert original is not None
    key = replace(original.key, context_identity='metadata-json-雪')
    descriptor = store.manifest(original.key.identity)
    assert descriptor is not None
    policy = {**descriptor['execution_policy'], 'test_metadata': {
        'unicode': '雪', 'boolean': True, 'integer': 7, 'null': None,
        'real': 1.25, 'negative_zero': -0.0,
        'nested': [{'values': [False, 0, 0.0, '', None]}],
    }}
    store.publish(NodeMaterialization(key, original.panel), execution_policy=policy)
    with store.read_transaction() as db:
        raw = db.execute('SELECT manifest_json FROM materializations WHERE identity=?', (key.identity,)).fetchone()[0]
    # JSON spelling and whitespace are not identity. Keep a noncanonical original
    # string so reserializing on insertion would fail the exact-cache assertion.
    raw = ' \n' + raw.replace('雪', '\\u96ea', 1) + '\n '
    with sqlite3.connect(store.meta_path) as db:
        db.execute('UPDATE materializations SET manifest_json=? WHERE identity=?', (raw, key.identity))
    return store, key.identity, raw, json.loads(raw)


def _forbid_values(reader, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Metadata read requested an artifact file, value decode or audit')
    for name in ('_file', '_decode', 'read_values', 'verify'):
        monkeypatch.setattr(reader, name, forbidden)


def test_cached_original_json_preserves_nested_types_bits_and_public_mutation_isolation(serialized, monkeypatch):
    store, identity, raw, expected = serialized
    assert len(raw.encode('utf-8')) > len(raw)
    token = _tree(expected)
    events = []
    with store.read_context(metadata_observer=lambda name, amount: events.extend([name] * amount)) as reader:
        _forbid_values(reader, monkeypatch)
        first = reader.manifest(identity)
        assert first is not None
        assert _tree(first) == token
        memo = reader._metadata_memo
        assert memo is not None
        assert memo.get(('manifest', identity), copy_value=False) == raw
        assert type(memo.get(('manifest', identity), copy_value=False)) is str
        for value in (first, reader.manifest(identity)):
            assert value is not None
            value['key']['inputs'].append('caller-only')
            value['execution_policy']['test_metadata']['nested'][0]['values'].clear()
            value['execution_policy']['test_metadata']['unicode'] = 'changed'
            assert _tree(reader.manifest(identity)) == token
        description = reader.describe(identity)
        assert description is not None
        description['key']['inputs'].append('description-only')
        description['execution_policy']['test_metadata']['nested'].clear()
        assert _tree(reader.manifest(identity)) == token
        assert events.count('manifest_reads') == 1
        assert events.count('manifest_cache_hits') >= 4
    assert memo.closed and memo.bytes == 0
    assert _tree(store.manifest(identity)) == token


def test_concurrent_hits_return_distinct_mutable_trees_after_context_closes(serialized, monkeypatch):
    store, identity, _raw, expected = serialized
    barrier = Barrier(4)
    with store.read_context() as reader:
        reader.manifest(identity)
        _forbid_values(reader, monkeypatch)

        def consume(marker):
            value = reader.manifest(identity)
            assert value is not None
            barrier.wait(timeout=10)
            value['key']['inputs'].append(marker)
            value['execution_policy']['test_metadata']['nested'][0]['values'].append(marker)
            return value

        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(consume, range(4)))
        assert len({id(value) for value in values}) == 4
        assert len({id(value['key']['inputs']) for value in values}) == 4
        assert _tree(reader.manifest(identity)) == _tree(expected)
    for marker, value in enumerate(values):
        assert value['key']['inputs'].pop() == marker
        assert value['execution_policy']['test_metadata']['nested'][0]['values'].pop() == marker
        assert _tree(value) == _tree(expected)


@pytest.mark.parametrize('capacity', ['zero', 'under', 'exact', 'large'])
def test_serialized_entry_keeps_existing_utf8_charge_admission_and_events(serialized, monkeypatch, capacity):
    store, identity, raw, expected = serialized
    charge = 1024 + 8 * len(raw.encode('utf-8'))
    budget = {'zero': 0, 'under': charge - 1, 'exact': charge, 'large': 16 * charge}[capacity]
    events = []
    with store.read_context(max_metadata_bytes=budget,
            metadata_observer=lambda name, amount: events.extend([name] * amount)) as reader:
        _forbid_values(reader, monkeypatch)
        for _ in range(3):
            assert _tree(reader.manifest(identity)) == _tree(expected)
        memo = reader._metadata_memo
        assert memo is not None
        admitted = budget >= charge
        assert memo.bytes == (charge if admitted else 0)
        assert memo.bytes <= memo.maximum_bytes
        assert events.count('manifest_reads') == (1 if admitted else 3)
        assert events.count('manifest_cache_hits') == (2 if admitted else 0)
        assert (memo.get(('manifest', identity), copy_value=False) == raw) is admitted
    assert memo.closed and memo.bytes == 0


@pytest.mark.parametrize('damage', ['json', 'identity'])
def test_failed_metadata_is_not_cached_and_restoration_is_revalidated(serialized, damage):
    store, identity, raw, expected = serialized
    corrupted = '{' if damage == 'json' else json.dumps({**expected, 'through': '2099-01-01'})
    with store.read_context() as reader:
        memo = reader._metadata_memo
        assert memo is not None
        assert reader.manifest('not-registered') is None
        with sqlite3.connect(store.meta_path) as db:
            db.execute('UPDATE materializations SET manifest_json=? WHERE identity=?', (corrupted, identity))
        with pytest.raises(ValueError):
            reader.manifest(identity)
        assert memo.get(('manifest', identity), copy_value=False) is None
        with sqlite3.connect(store.meta_path) as db:
            db.execute('UPDATE materializations SET manifest_json=? WHERE identity=?', (raw, identity))
        assert _tree(reader.manifest(identity)) == _tree(expected)
        assert memo.get(('manifest', identity), copy_value=False) == raw


@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_json_metadata_keeps_existing_publication_rejection(serialized, nonfinite):
    store, identity, _raw, expected = serialized
    original = store.read(identity)
    assert original is not None
    key = replace(original.key, context_identity='rejected-nonfinite')
    policy = {**expected['execution_policy'], 'nonfinite': nonfinite}
    with pytest.raises(ValueError, match='Out of range float values'):
        store.publish(NodeMaterialization(key, original.panel), execution_policy=policy)
    assert store.manifest(key.identity) is None
