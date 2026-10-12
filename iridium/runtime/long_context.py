"""Bounded exact KV plus learned compressed context; no unlimited KV growth."""
import torch

from .decode import slice_batch


class LongContextSession:
    """One isolated stream. Supports up to a declared ingestion budget.

    Raw KV is reset at local-window boundaries; learned memory persists. This
    intentionally differs from full attention. No guarantee of exact recall.
    Do not reuse a session across users, documents, or modified model weights.
    """
    def __init__(self, model, n_loops=3, window=None, max_tokens=1_048_576):
        if model.context_memory is None:
            raise ValueError('enable memory_slots and train memory before using this path')
        self.model = model
        self.window = window or model.cfg.max_seq_len
        if not 1 <= self.window <= model.cfg.max_seq_len or max_tokens < self.window:
            raise ValueError('invalid local window or total ingestion budget')
        if not 1 <= n_loops <= model.cfg.router.max_loops:
            raise ValueError('invalid deliberation budget')
        self.n_loops, self.max_tokens = n_loops, max_tokens
        self.reset()

    def reset(self):
        self.cache = {}
        self.seen = 0
        self.last_hidden = None

    @torch.no_grad()
    def consume(self, batch):
        if self.model.training:
            raise ValueError('call model.eval() before inference')
        if batch.shape[0] != 1 or not bool(batch.valid.all()):
            raise ValueError('session requires a single unpadded stream')
        length = batch.shape[1]
        if length < 1 or self.seen + length > self.max_tokens:
            raise ValueError('empty chunk or total ingestion budget exceeded')
        start = 0
        while start < length:
            occupied = self.cache.get(('stream', 'n'), 0)
            if occupied == self.window:
                memory = self.cache.get(('context', 'state'))
                self.cache = {('context', 'state'): memory}
                occupied = 0
            end = min(length, start + self.window - occupied)
            for _, first, shape in batch.grids:
                count = 1
                for dimension in shape:
                    count *= dimension
                if first < end < first + count or first < start < first + count:
                    raise ValueError('local boundary splits an atomic field grid; choose another window')
            chunk = slice_batch(batch, start, end)
            chunk.positions = torch.arange(self.seen, self.seen + end - start,
                                           device=batch.positions.device)[None]
            try:
                output = self.model(chunk, n_loops=self.n_loops, cache=self.cache)
            except Exception:
                self.reset()  # partial layer-cache updates cannot be replayed safely
                raise
            self.last_hidden = output.hidden[:, -1:].detach()
            self.seen += end - start
            start = end
        return self.last_hidden

    def storage_report(self):
        def size(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (tuple, list)):
                return sum(size(v) for v in value)
            if hasattr(value, '__dict__'):
                return size(vars(value))
            return 0
        return {'tokens_ingested': self.seen, 'exact_local_tokens': self.cache.get(('stream', 'n'), 0),
                'cache_tensor_bytes': size(self.cache), 'compressed_context_is_lossy': True}

    @torch.no_grad()
    def continue_text(self, sample, max_new_tokens=128):
        """Append a prompt to this stream and decode native text greedily."""
        from ..codecs.bank import TensorBatch
        from ..codecs.spans import collate, MODALITY_INDEX
        from .generate import _single_token_batch
        from .chat import EOS
        if max_new_tokens < 1 or self.seen + len(sample) + max_new_tokens > self.max_tokens:
            raise ValueError('response would exceed the stream budget')
        batch = TensorBatch(collate([sample], self.model.cfg.codecs.continuous_dims(),
                                    self.model.cfg.codecs.action_scalars),
                            device=next(self.model.parameters()).device)
        hidden = self.consume(batch)
        output = bytearray()
        for _ in range(max_new_tokens):
            token = int(self.model.codecs.text_head(hidden)[0, -1].argmax())
            step = _single_token_batch(batch, MODALITY_INDEX['text'], token, self.seen)
            hidden = self.consume(step)
            if token == EOS:
                break
            if 16 <= token < 272:
                output.append(token - 16)
        return output.decode('utf-8', errors='replace')
