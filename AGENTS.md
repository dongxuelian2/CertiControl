# Repository instructions

## Zero-defensive-prose rule — mandatory

Apply this rule to all repository-authored English prose: README, mathematical research notes, `PROOF.md`, LaTeX exposition, documentation, pull-request descriptions, and code comments. Enforce it when auditing historical material and when writing anything new.

**Prohibited rhetorical constructions (including grammatical variants):**
- Chinese examples to recognize and eliminate in translated source material: `不是……`, `并不是……`, `不构成……`, `还没有做……`, `尚未……`, `而不是……`, `并未……`, `不意味着……`, `不代表……`, `不能据此……`, `这并不证明……`, `我们不声称……`, `不保证……`, `仅仅是……`.
- English equivalents to eliminate from authored explanations: `not a ...`, `is not ...`, `does not ...`, `do not ...`, `has not ...`, `have not ...`, `not yet ...`, `rather than ...`, `instead of ...`, `does not constitute ...`, `does not establish ...`, `does not imply ...`, `should not be interpreted as ...`, `we do not claim ...`, `no claim is made ...`, `merely ...`, `only a preliminary ...`, `beyond the scope ...`, `not intended to ...`.
- Delete disclaimers, self-exculpation, premature qualification, repetitive status caveats, and defensive contrasts. Do not relocate them to a different paragraph or replace them with synonyms serving the same rhetorical purpose. Zero such constructions are acceptable in explanatory prose.

**Required replacement style:** Lead with the actual theorem, deliverable, tested behavior, evidence, applicability conditions, exact scope, and concrete open mathematical question, using affirmative statements. Examples:
- `We prove the classification for n >= N under hypotheses H1 and H2.`
- `Verified cases: n = 2,...,7. General-case proof target: ...`
- `CI checks: 16 passed. Proof: Proposition 4. Research priority: external review.`
- `Algorithm: FPT for disjoint ordinary cardinality quotas. Next target: robust mixed-status source-SCC accessibility.`

**Mathematical truth takes precedence over stylistic cleanup.** Preserve essential logical negation in theorems, inequalities, counterexamples, constraints, source quotations, and executable code where changing it would alter correctness. In ordinary explanation, express boundaries by positive quantified scope and precise next objectives. If a claim is unsupported, correct or remove it; never invent evidence to avoid a negative sentence.

Before opening or updating a PR, review every added or modified paragraph for prohibited rhetoric and its semantic equivalents. Record the direct result and evidence; keep mathematical limitations in compact, exact statements of domain and open problems.

Preserve mathematical and empirical evidence accurately while keeping the project straightforward to reproduce.

## Mandatory English and evidence review

- English is the sole language for repository-authored prose, including README, AGENTS, Markdown research notes and proof explanations, LaTeX exposition, code comments/docstrings, and PR descriptions. Audit and translate existing Chinese passages across tracked files, nested directories, and historical materials. Preserve mathematical notation, exact technical meaning, citations, and essential original-language titles/quotations with English context.
- Review every existing technical or research claim and its evidence. Check assumptions, quantifiers, proof obligations, references, test coverage, computation-to-claim consistency, numerical validity and reproducibility. Supply missing definitions and valid arguments, correct factual errors and ambiguous assertions, and accurately distinguish formal proofs, analytical arguments, finite experiments, simulations, and open conjectures.
- Where a problem cannot be resolved, identify the exact unresolved claim and narrow its status; never invent proof steps or silently exaggerate results. Preserve necessary scripts and functional behavior. Avoid unnecessary boilerplate or engineering scaffolding.
- Propose self-contained, reviewable PRs per repository for the legacy audit. Recheck English consistency, internal links and equations, verify mathematical equivalence of any translation, and run relevant existing tests. Enforce these requirements on all future additions and updates.

