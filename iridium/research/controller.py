"""Trusted orchestration for configuration-only research on notebook hosts.

Processes provide memory/lifetime separation, NOT a hostile-code sandbox.
The model has only a bounded JSON proposal interface. Code patches are proposals
for human review and never reach an execution primitive. Policy, evaluator and
dataset digests are pinned by this controller, not supplied by the model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .policy import ResearchPolicy, judge, validate_proposal


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def version_digest(path):
    return hashlib.sha256((digest(Path(path)/'weights.pt') + digest(Path(path)/'model.json')).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def manifest_inventory(path, split):
    path = Path(path).resolve()
    records, files = [], {str(path): digest(path)}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('split') != split:
            continue
        if not all(record.get(k) for k in ('id', 'source', 'license')):
            raise ValueError('every research example needs id, recording/source identity and license')
        records.append(record)
        for turn in record['turns']:
            for part in turn['content']:
                if 'path' not in part:
                    continue
                media = (path.parent / part['path']).resolve()
                if not media.is_relative_to(path.parent) or not media.is_file():
                    raise ValueError('media must exist within its manifest directory')
                files[str(media)] = digest(media)
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('empty split or duplicate example IDs')
    return records, files


class ResearchLoop:
    """Initialize once per bounded campaign; no background scheduler is installed."""
    def __init__(self, directory, train_manifest, selection_manifest, audit_manifest,
                 policy=None, device='cuda:0'):
        self.root = Path(directory).resolve()
        if self.root.exists():
            raise FileExistsError('use a new campaign directory; prior audit/version records are retained')
        self.root.mkdir(parents=True)
        self.policy = policy or ResearchPolicy()
        self.device = device
        self.paths = {key: str(Path(value).resolve()) for key, value in
                      [('train', train_manifest), ('selection', selection_manifest), ('audit', audit_manifest)]}
        self.records, self.pinned, media_digests = {}, {}, {}
        for split, path in self.paths.items():
            records, files = manifest_inventory(path, split)
            self.records[split] = records
            self.pinned.update(files)
            media_digests[split] = {value for key, value in files.items() if key != path}
        for left, right in [('train', 'selection'), ('train', 'audit'), ('selection', 'audit')]:
            if media_digests[left] & media_digests[right]:
                raise ValueError(f'identical media content across {left} and {right}')
            for key in ('id', 'source'):
                if {r[key] for r in self.records[left]} & {r[key] for r in self.records[right]}:
                    raise ValueError(f'{key} overlap between {left} and {right}; split by original source')
        for split in ('selection', 'audit'):
            sampled = self.records[split][:self.policy.eval_items]
            available = {row.get('family', 'multimodal') for row in sampled}
            required = {row.get('family', 'multimodal') for row in self.records['train'][:self.policy.max_items]}
            if not required <= available:
                raise ValueError(f'{split} is missing training family coverage within eval_items')
        self.families = sorted({r.get('family', 'multimodal') for r in self.records['train']})
        # Snapshot only trusted Python. Workers cannot substitute candidate source code.
        source = Path(__file__).resolve().parents[2]
        self.source = self.root / 'trusted_source'
        for path in (source / 'iridium').rglob('*.py'):
            destination = self.source / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            self.pinned[str(destination)] = digest(destination)
        atomic_json(self.root/'policy.json', self.policy.to_dict())
        self.pinned[str(self.root/'policy.json')] = digest(self.root/'policy.json')
        atomic_json(self.root/'pins.json', self.pinned)
        self.audit = []
        self.deadline = None

    def _check_pins(self):
        for path, expected in self.pinned.items():
            if not Path(path).is_file() or digest(path) != expected:
                raise RuntimeError(f'protected evaluator/policy/data changed: {path}')

    def initialize(self, trusted_checkpoint):
        """Import user's trusted checkpoint once; worker weights use weights_only=True."""
        from ..training.trainer import load_checkpoint
        from .worker import save_version
        self._check_pins()
        model, manifest = load_checkpoint(trusted_checkpoint, 'cpu')
        if model.cfg.n_params > self.policy.max_parameters:
            raise ValueError('incumbent exceeds campaign parameter budget')
        parent = self.root / 'versions/v000'
        if parent.exists():
            raise FileExistsError('campaign already initialized')
        save_version(model, parent, manifest.get('train_config', {}).get('n_loops', 1),
                     {'initial_checkpoint_sha256': digest(trusted_checkpoint)})
        self.current = parent
        atomic_json(self.root/'current.json', {'version': str(parent), 'sha256': version_digest(parent)})
        del model
        return parent

    def _job(self, kind, parent, *, proposal=None, split=None, output=None, evidence=None, seed=None):
        self._check_pins()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('campaign wall-clock budget exhausted')
        folder = self.root / 'jobs' / uuid.uuid4().hex
        folder.mkdir(parents=True)
        job = {'kind': kind, 'parent': str(parent), 'device': self.device,
               'policy': self.policy.to_dict(), 'result': str(folder/'result.json'),
               'seed': self.policy.seed if seed is None else seed,
               'families': self.families}
        if kind == 'train':
            job.update(proposal=proposal, manifest=self.paths['train'], output=str(output))
        if kind == 'evaluate':
            job.update(manifest=self.paths[split], split=split)
        if kind == 'propose':
            job['evidence'] = evidence
        atomic_json(folder/'job.json', job)
        before = version_digest(parent)
        # The command is constant. No generated text enters -c or a shell.
        command = [sys.executable, '-I', '-B', '-c',
                   'import sys,runpy;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("iridium.research.worker",run_name="__main__")',
                   str(self.source), str(folder/'job.json')]
        with (folder/'worker.log').open('wb') as log:
            subprocess.run(command, cwd=self.source, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(remaining, self.policy.job_seconds))
        self._check_pins()
        if before != version_digest(parent):
            raise RuntimeError('worker modified its parent version')
        return json.loads((folder/'result.json').read_text(encoding='utf-8'))['result']

    def run(self, proposals=None):
        """Run a finite campaign. Each round contains an unchanged-config control.

        ``proposals`` may be a human-authored list for weak initial models. When
        omitted, the CURRENT promoted model proposes each non-control candidate.
        Bad proposals fail explicitly; no fabricated researcher output is substituted.
        """
        if not hasattr(self, 'current'):
            raise RuntimeError('initialize with a trained checkpoint first')
        if self.deadline is not None:
            raise RuntimeError('campaign already run; create a new campaign to change its budget')
        self.deadline = time.monotonic() + self.policy.max_wall_seconds
        if proposals is not None and (not isinstance(proposals, list) or len(proposals) != self.policy.candidates-1):
            raise ValueError('provide exactly candidates-1 proposals; candidate 0 is the control')
        summary = {'last': 'initial checkpoint'}
        try:
            for round_index in range(self.policy.rounds):
                parent = self.current
                parent_hash = version_digest(parent)
                descriptor = json.loads((parent/'model.json').read_text(encoding='utf-8'))
                baseline = self._job('evaluate', parent, split='selection')
                summary = {'parent_loops': descriptor['n_loops'],
                           'controller_mode': descriptor['config'].get('controller_mode', False),
                           'text_accuracy': round(baseline['text_accuracy'], 4),
                           'mean_loss': round(sum(sum(x['losses'].values()) / max(1, len(x['losses']))
                                                 for x in baseline['examples']) / len(baseline['examples']), 4),
                           'last': summary.get('last_selection_gain', 'initial'),
                           'gated_bank': descriptor['config'].get('gated_bank', False)}
                passed = []
                for candidate_index in range(self.policy.candidates):
                    record = {'round': round_index, 'candidate': candidate_index, 'parent_sha256': parent_hash}
                    output = self.root/f'candidates/r{round_index:03}-c{candidate_index:03}'
                    try:
                        raw = {k: v for k, v in descriptor.get('provenance', {}).get('proposal', {}).items() if k != 'code_patch'} if candidate_index == 0 else (proposals[candidate_index-1] if proposals is not None else
                            self._job('propose', parent, evidence=summary,
                                      seed=self.policy.seed+round_index*100+candidate_index))
                        proposal = validate_proposal(raw, self.families, descriptor['config']['router']['max_loops'])
                        record['proposal'] = proposal
                        if proposal.get('code_patch'):
                            review = self.root/'code_proposals'; review.mkdir(exist_ok=True)
                            (review/f'r{round_index}-c{candidate_index}.patch.txt').write_text(proposal['code_patch'], encoding='utf-8')
                            record.update(status='review_only', reason='Python changes are not executed by the notebook backend')
                            continue
                        self._job('train', parent, proposal=proposal, output=output,
                                  seed=self.policy.seed+round_index)
                        candidate_hash = version_digest(output)
                        metrics = self._job('evaluate', output, split='selection')
                        verdict = judge(baseline, metrics, self.policy)
                        record.update(status='evaluated', selection=verdict, sha256=candidate_hash)
                        if verdict['passed']:
                            passed.append((verdict['lower_bound'], output, candidate_hash))
                    except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                        record.update(status='failed', error=str(exc))
                    finally:
                        self.audit.append(record)
                        atomic_json(self.root/'audit.json', self.audit)
                if not passed:
                    break
                _, winner, expected_hash = max(passed, key=lambda row: row[0])
                # Only the selected finalist sees the final audit phase. It cannot
                # modify the evaluator; audit answers/metrics never enter its prompt.
                audit_parent = self._job('evaluate', parent, split='audit')
                audit_candidate = self._job('evaluate', winner, split='audit')
                verdict = judge(audit_parent, audit_candidate, self.policy)
                if not verdict['passed']:
                    self.audit.append({'round': round_index, 'status': 'audit_rejected', 'verdict': verdict})
                    break
                self._check_pins()
                if version_digest(winner) != expected_hash or version_digest(parent) != parent_hash:
                    raise RuntimeError('checkpoint changed after evaluation')
                promoted = self.root / f'versions/v{round_index+1:03}'
                shutil.copytree(winner, promoted)
                if version_digest(promoted) != expected_hash:
                    raise RuntimeError('promotion copy differs from evaluated artifact')
                atomic_json(self.root/'current.json', {'version': str(promoted), 'sha256': expected_hash,
                                                      'previous': str(parent)})
                self.current = promoted
                self.audit.append({'round': round_index, 'status': 'promoted', 'version': str(promoted),
                                   'sha256': expected_hash, 'audit_verdict': verdict})
                summary = {'last_selection_gain': max(passed, key=lambda row: row[0])[0],
                           'training_steps_per_candidate': self.policy.train_steps}
        finally:
            atomic_json(self.root/'audit.json', self.audit)
        return {'current': str(self.current), 'sha256': version_digest(self.current),
                'audit': str(self.root/'audit.json')}
