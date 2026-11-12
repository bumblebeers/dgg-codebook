# dgg-codebook: Continuous Vocabulary Distribution Derivation, De-Biasing, and Spatial Teachers

`dgg-codebook` provides the mathematical foundation and neural models for treating DiffusionGemma's vocabulary as a continuous overcomplete dictionary rather than a discrete token bottleneck.

## Core Concepts

1. **Continuous Vocabulary Basis**:
   Rather than collapsing each spatial patch into a single discrete token ID (which induces severe mode collapse and starves downstream heads of continuous bandwidth), each patch is supervised against a **multi-modal probability distribution $q_i(v)$** over valid vocabulary tokens.
2. **Continuous Vocabulary Chords ($e_{\text{soft}}$)**:
   The expected embedding $e_i = \sum_k \tilde{q}_{i, k} E_{\text{lm}}[v_k] \in \mathbb{R}^{2816}$ forms a continuous linear combination ("chord") of vocabulary vectors providing the degrees of freedom required to decode high-fidelity visual latents.
3. **Statistical Codebook De-Biasing**:
   - Frequency discounting to prevent vocabulary domination by common tokens.
   - Background suppression to preserve high-frequency edge and boundary salience.
   - Positional RoPE carrier de-biasing: Subtracting the prompt-invariant 1D RoPE wave $\mu_{\text{pos}}[0..255]$ from decoder states to eliminate visual regression to the mean.
4. **Spatial Teacher Distillation**:
   Compact surrogate architectures (`IsoSpatialTeacher` and `CtxSpatialTeacher`) distilling 26B multi-prompt elicitation into fast local photometric and contextual topological codebooks.

## Installation

```bash
pip install git+https://github.com/bumblebeers/dgg-codebook.git
# Or for local editable development:
git clone https://github.com/bumblebeers/dgg-codebook.git && cd dgg-codebook && pip install -e .[test]
```

## Related Resources

- **Main Research & Benchmark Repository**: [bumblebeers/dgg2-latent-code-graphs](https://github.com/bumblebeers/dgg2-latent-code-graphs)
- **Pretrained Adapters & ImageDecoder Weights**: [bumblebeer/dgg2-latent-code-graphs-adapters](https://huggingface.co/bumblebeer/dgg2-latent-code-graphs-adapters)
- **AST Code-Graph Benchmark Packs**: [bumblebeer/dgg2-code-graph-benchmarks](https://huggingface.co/datasets/bumblebeer/dgg2-code-graph-benchmarks)
- **License**: Apache-2.0
