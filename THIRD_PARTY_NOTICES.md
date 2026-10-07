# Third-party software, models, and data

The MIT license in the repository root applies to the original JevNexus code and
documentation. It does not replace the licenses of third-party software,
downloaded model weights, or benchmark data.

## Vendored software

`vendor/magneto/` contains a pinned and minimally modified copy of Magneto from
[`VIDA-NYU/magneto-matcher`](https://github.com/VIDA-NYU/magneto-matcher),
commit `6620623265fc7feac0f053996e62b68a13a72a57`. Magneto is licensed under
Apache License 2.0. The included license is at `vendor/magneto/LICENSE`, and the
local changes are documented in `vendor/magneto/PATCHES.md`.

Other Python packages are installed separately and remain subject to their own
licenses.

## Downloaded models

Model weights are downloaded at runtime and are not distributed in this
repository. At the pinned or documented versions used by JevNexus:

| Model | Purpose | Reported license |
|---|---|---|
| [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | Candidate retrieval | Apache-2.0 |
| [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) | Magneto-Qwen reranking and Open-Jev base model | Apache-2.0 |
| [`ZefanCai/Open-Jev-9B`](https://huggingface.co/ZefanCai/Open-Jev-9B) | Decision model adapter and head | Apache-2.0 weights; MIT code |
| [`jinaai/jina-reranker-v3.5`](https://huggingface.co/jinaai/jina-reranker-v3.5) | JevNexus listwise refinement | CC BY-NC 4.0 |
| [`RUC-DataLab/unicorn-plus-v1`](https://huggingface.co/RUC-DataLab/unicorn-plus-v1) | Unicorn baseline checkpoint | MIT |
| [`microsoft/deberta-base`](https://huggingface.co/microsoft/deberta-base) | Unicorn encoder | MIT |

The Jina reranker is non-commercial. Users are responsible for reviewing the
current upstream model terms before use, especially in commercial settings.

## Benchmark data

JevNexus does not redistribute benchmark archives. `dema.data.download` retrieves
them from their official records:

| Dataset | Record | Reported license |
|---|---|---|
| GDC-SM | [Zenodo 14963588](https://doi.org/10.5281/zenodo.14963588) | CC BY 4.0 |
| Valentine Datasets | [Zenodo 5084605](https://doi.org/10.5281/zenodo.5084605) | CC BY 4.0 |

The dataset references and layout follow the public Magneto benchmark setup.
Downloaded data remains subject to its upstream terms and attribution
requirements.
