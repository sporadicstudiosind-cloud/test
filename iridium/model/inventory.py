"""Executable parameter and cache inventory.

Architecture §19 supplies configurations and formulae. §19.1 says an executable
inventory over the actual implementation is the final authority, so this module
implements the formulae and the test suite checks them against both the
document's stated counts and — for the prototype — the real ``torch`` module.
"""

from __future__ import annotations

from dataclasses import dataclass

MIB = 1024 ** 2
TIB = 1024 ** 4


@dataclass(frozen=True)
class TransformerConfig:
    name: str
    d_model: int
    n_prelude: int
    n_core: int
    n_coda: int
    d_ff: int
    n_query_heads: int
    n_kv_heads: int
    d_head: int

    def __post_init__(self) -> None:
        if self.d_model != self.n_query_heads * self.d_head:
            raise ValueError(
                f"{self.name}: d_model {self.d_model} != "
                f"n_query_heads {self.n_query_heads} * d_head {self.d_head}"
            )
        if self.n_query_heads % self.n_kv_heads:
            raise ValueError(f"{self.name}: query heads must be a multiple of kv heads")

    @property
    def d_kv(self) -> int:
        return self.n_kv_heads * self.d_head

    @property
    def n_blocks(self) -> int:
        return self.n_prelude + self.n_core + self.n_coda

    @property
    def params_per_block(self) -> int:
        """``2 d^2 + 2 d d_kv + 3 d d_ff``.

        Q and O projections are ``d x d``; K and V are ``d x d_kv``; SwiGLU uses
        three ``d x d_ff`` matrices. Biases and norms are excluded and counted
        separately, as §19.1 requires.
        """
        d, d_kv, d_ff = self.d_model, self.d_kv, self.d_ff
        return 2 * d * d + 2 * d * d_kv + 3 * d * d_ff

    @property
    def transformer_params(self) -> int:
        return self.n_blocks * self.params_per_block

    def kv_bytes_per_token(self, recurrence: int, bytes_per_element: int = 2) -> int:
        """Cache size for one token with the recurrence-indexed cache design.

        Architecture §6.5 keys the cache by recurrence index, so a core layer
        run ``R`` times occupies ``R`` cache slots rather than one.
        """
        stages = self.n_prelude + recurrence * self.n_core + self.n_coda
        return 2 * self.d_kv * bytes_per_element * stages

    def kv_bytes_for_stream(
        self, tokens: int, recurrence: int, bytes_per_element: int = 2
    ) -> int:
        return tokens * self.kv_bytes_per_token(recurrence, bytes_per_element)

    def weight_bytes(self, bytes_per_param: int = 2) -> int:
        return self.transformer_params * bytes_per_param

    def training_state_bytes(self, bytes_per_param: int = 16) -> int:
        """BF16 params + BF16 grads + FP32 master + two FP32 moments ≈ 16 B/param."""
        return self.transformer_params * bytes_per_param


PROTOTYPE = TransformerConfig(
    name="iridium-1-prototype",
    d_model=1024,
    n_prelude=2,
    n_core=8,
    n_coda=2,
    d_ff=2816,
    n_query_heads=16,
    n_kv_heads=4,
    d_head=64,
)

PILOT = TransformerConfig(
    name="iridium-1-pilot",
    d_model=4096,
    n_prelude=4,
    n_core=24,
    n_coda=4,
    d_ff=11264,
    n_query_heads=32,
    n_kv_heads=8,
    d_head=128,
)

FLAGSHIP = TransformerConfig(
    name="iridium-1-flagship",
    d_model=32768,
    n_prelude=4,
    n_core=80,
    n_coda=4,
    d_ff=90112,
    n_query_heads=256,
    n_kv_heads=32,
    d_head=128,
)

CONFIGS = {c.name: c for c in (PROTOTYPE, PILOT, FLAGSHIP)}


def report(config: TransformerConfig, recurrences: tuple[int, ...] = (1, 4)) -> str:
    lines = [
        f"{config.name}",
        f"  d_model            {config.d_model:,}",
        f"  blocks             {config.n_prelude}/{config.n_core}/{config.n_coda}"
        f" = {config.n_blocks}",
        f"  d_kv               {config.d_kv:,}",
        f"  params/block       {config.params_per_block:,}",
        f"  transformer params {config.transformer_params:,}",
        f"  BF16 weights       {config.weight_bytes() / 1e12:.3f} TB (decimal)",
        f"  training states    {config.training_state_bytes() / 1e12:.3f} TB (decimal)",
    ]
    for r in recurrences:
        per_token = config.kv_bytes_per_token(r)
        per_stream = config.kv_bytes_for_stream(1_048_576, r)
        lines.append(
            f"  KV @ R={r:<2d}         {per_token / MIB:.3f} MiB/token, "
            f"{per_stream / TIB:.3f} TiB per 1,048,576-token stream"
        )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    for cfg in CONFIGS.values():
        print(report(cfg))
        print()
