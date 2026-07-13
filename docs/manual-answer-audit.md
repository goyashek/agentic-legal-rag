# Manual audit of saved answers

I compared the final baseline and full-graph answers with their cited text in
`data/processed/sections.jsonl`. This is a targeted ten-case audit, not another
headline metric. It covers all three hard cases from the 20-case run, four medium
cases, and three easy cases where the full graph ended low confidence.

An answer passes when it identifies the central provision, supports it with the
cited text, and does not make a material mistake about punishment or procedure.
"Partial" means that the core is useful but a needed provision is missing or a
side claim is unsupported. "Fail" means that the final response did not answer
the question.

| scenario | baseline | full graph | note from the statute check |
|---|---|---|---|
| s48 armed rebellion material | pass | pass | BNS 152 directly covers the central conduct. |
| s38 finder keeps wallet | partial | pass | Both identify BNS 314. Baseline states the wrong punishment options; the section requires at least six months and a fine. |
| s50 attempted chain snatching | partial | partial | Both correctly say completed snatching needs seizure, but neither cites BNS 62 on attempts. |
| s07 threat to kill to stop a complaint | pass | fail | BNS 351 supports the baseline. The full graph discarded three citation-valid attempts and returned a generic reply. |
| s43 burying a body | pass | pass | BNS 238 and its illustration support both answers. |
| s05 dowry harassment and beatings | pass | partial | Baseline identifies BNS 85 and its punishment. Full stops at BNS 86, the definition, and omits the offence section. |
| s28 false evidence under oath | partial | fail | Baseline identifies BNS 227 but misses BNS 229, the ordinary punishment provision. The full graph returned no answer. |
| s19 unwanted advances at work | partial | fail | BNS 75 supports the core answer. The extra BNS 68 and BNS 78 discussion needs facts not supplied in the query. |
| s25 permanent loss of sight after beating | partial | fail | BNS 116 correctly classifies the injury, but BNS 117 is the needed offence and penalty section. |
| s03 fake gold chain sale | pass | fail | BNS 318 supports the baseline's central cheating analysis. The full graph returned no answer. |

The baseline has five passes and five partial answers. It answers all ten
queries. The full graph has three passes, two partial answers, and five failures.
All five full-graph failures came after the checker rejected citation-valid
generated answers and exhausted the rewrite budget.

The audit agrees with the 20-case metrics: the checker and rewrite loop are
rejecting useful, grounded answers often enough to hurt the result. This does
not mean the baseline is safe by itself. The s38 error shows that citation
membership is not claim-level checking. The live path should keep scope controls,
the exact-section lookup, dense retrieval, and deterministic citation validation.
It should not run the grader, checker, or rewriter on every normal in-corpus
question. The generator prompt now requires exact punishment bounds and whether
a fine is mandatory or optional, and a key-free regression test covers the BNS
314 source text. The simpler path is now the live default.

## Retrieval follow-up

The three missing provisions in this audit were retrieval-window failures, not
index misses. A local dense check with `BAAI/bge-large-en-v1.5` and the rebuilt
corpus ranked BNS 117 at 11 for s25, BNS 229 at 12 for s28, and BNS 62 at 10 for
s50. BNS 304 was eighth for s50. The live path then kept only eight chunks for
generation.

The live context window is now 12 chunks. Key-free graph tests verify that cap,
and local integration tests verify that all three scenarios reach the generator
context. This is a three-query retrieval check, not a new RAGAS result and it
used no LLM calls.
