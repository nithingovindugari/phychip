# Data and license notice

## Release license split

- `phy-chip-env` framework, harnesses, and authored tests: **Apache-2.0**.
- PhyChip model adapter and released datasets: **CC-BY-NC-SA-4.0**,
  educational/research, non-commercial.
- ngspice: external system dependency under its own upstream terms; not bundled
  in the Python wheel.
- SmolLM3 base weights: retain their upstream Hugging Face license and terms.

## Exact clean-SFT corpus

The v0.1 model was trained only from
`data/final_data/sft_v0_verified_v1/sft_training_ready.jsonl`:

- rows: 15,660;
- SHA-256: `02af3e6c58b668acec130850ea31d468fb8fd49f938272854d4d857b0fd39b9b`;
- four project-controlled synthetic-authoring buckets recorded by the private
  dynamic manifest: 7,359, 4,849, 2,888, and 564 rows;
- no rows from the blocked benchmark-contamination source.

Every row retains source-file, source-line, row, response/netlist, authoring-
stream, and gate identities in the private accompanying manifests. Public
release documents intentionally use provider-neutral provenance. The release-level
CC-BY-NC-SA-4.0 posture is intentionally conservative and must travel with any
redistribution or derivative adapter.

The broad `athma-train/NOTICE` is a research inventory, not the exact lineage
notice for these weights. In particular, its unresolved `goihere/Netlist_data`,
`hdl2v/vhdl-dataset`, and `HuggingFaceTB/smoltalk` entries must not be inferred
to be part of the clean-SFT corpus from that inventory alone.

## Training-data claim boundary

“Verified” means static-clean, assistant-mask-valid, sandbox-admitted, and
ngspice-46 simulator-passing. It does not mean every row meets every numeric
specification, uses the intended topology, or contains a unique topology.

## Redistribution obligations

Redistributors must retain attribution, the CC-BY-NC-SA-4.0 notice, the
non-commercial limitation, and share adaptations under the same license. This
summary is not legal advice; consult the full license text distributed as
`LICENSE_WEIGHTS_DATA.txt`.
