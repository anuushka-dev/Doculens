# Phase 2 Retrieval Dataset Review

**Review date:** 2026-10-29
**Branch/review baseline:** `phase2`, commit `0d072885b8be8ebd44367b47c5c544345f32865d`
**Dataset:** `tests/phase2/fixtures/retrieval_eval.jsonl`
**Source:** `sample_docs/nist_sp_800_63_4.pdf`, NIST SP 800-63-4, *Digital Identity Guidelines* (96 physical PDF pages). The rendered-source cross-check used the official final NIST PDF: [NIST SP 800-63-4 PDF](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63-4.pdf).

## Scope and method

All 30 question rows were reviewed against the text and rendered view of each labelled source page. Page references in the fixture are **one-based physical PDF page numbers**, not the printed footer numbers; front matter makes those numbers differ. Review checked that each target page contains direct evidence answering its query, not merely a matching keyword.

This is a single-reviewer source audit, not an inter-annotator agreement study. The review does not claim that the retrieval benchmark measures answer-generation correctness. No additional page-label corrections were needed beyond the two evidence-based expansions already applied to Q001 and Q025.

## Row-by-row findings

| ID | Labelled physical PDF page(s) | Evidence verified on the rendered page | Disposition |
| --- | --- | --- | --- |
| Q001 | `[1, 2]` | Both the cover and formal title page show `NIST SP 800-63-4` and `Digital Identity Guidelines`. | Retain both pages; label expansion justified. |
| Q002 | `[5]` | The abstract states the three areas: identity proofing, authentication, and federation. | Label verified. |
| Q003 | `[8]` | The table of contents lists section 3.3.3 as “Initial Assurance Level Selection.” | Label verified. |
| Q004 | `[11]` | The introduction says a digital identity is always unique within the context of an online service. | Label verified. |
| Q005 | `[14]` | The text describes SP 800-63B requirements for authentication processes at each AAL, including authenticator choices and their lifecycle. | Label verified. |
| Q006 | `[18]` | The notations section explains that SHALL/SHALL NOT indicate strict requirements with no permitted deviation. | Label verified. |
| Q007 | `[21]` | The text lists “something you know,” “something you have,” and “something you are.” | Label verified. |
| Q008 | `[24]` | The federation benefits list describes minimizing data exposed to RPs through pseudonymous identifiers and derived attribute values. | Label verified. |
| Q009 | `[27]` | Step 1 of the general-purpose federated model is application to a CSP through identity proofing and enrollment. | Label verified. |
| Q010 | `[30]` | Step 5 of the subscriber-controlled wallet flow says the claimant proves possession and control of the wallet. | Label verified. |
| Q011 | `[33]` | The first DIRM process step is “Define the online service,” including scope, users, transactions, underlying data, and impacted entities. | Label verified. |
| Q012 | `[36]` | The page contains the explicit minimum list RPs must include in an online-service description. | Label verified. |
| Q013 | `[39]` | The page lists harms from unauthorized access to information, including sensitive-data exposure and financial, physical, psychological, impersonation, identity-theft, and inconvenience harms. | Label verified. |
| Q014 | `[42]` | The water-treatment example explains analysing each user group separately based on available transactions and considering affected entities. | Label verified. |
| Q015 | `[45]` | The IAL summary states IAL1 aims to limit highly scalable attacks and protect against synthetic identity and compromised personal information. | Label verified. |
| Q016 | `[49]` | The text says to document whether identity proofing is needed and select initial IAL per user group from the effective impact-level determination. | Label verified. |
| Q017 | `[52]` | The page lists the governance approach, documented tailoring decisions and controls in the DIAS, and risk-based justifications. | Label verified. |
| Q018 | `[55]` | The DIAS section defines its purpose and lists minimum contents including impact assessment, xAL selection/rationale, compensating controls, and supplemental controls. | Label verified. |
| Q019 | `[58]` | The performance-metrics table defines overall fail rate as the percentage of unique users who start proofing but cannot complete all steps. | Label verified. |
| Q020 | `[61]` | The redress section requires a user-accessible issue-handling process for grievances, evidence review, additional evidence, resolution, and corrective action. | Label verified. |
| Q021 | `[65]` | The references list gives the NIST SP 800-37r2 title, “Risk Management Framework for Information Systems and Organizations: A System Life Cycle Approach for Security and Privacy.” | Label verified. |
| Q022 | `[69]` | The abbreviations list defines NFC (communications protocol) as Near-Field Communication, distinct from NFC for Unicode normalization. | Label verified; query is appropriately disambiguated. |
| Q023 | `[74]` | The glossary defines an attribute bundle as a package of attribute values and derived attribute values from a CSP, protected for validation. | Label verified. |
| Q024 | `[77]` | The glossary defines back-channel communication as a direct system-to-system connection without redirects through an intermediary such as a browser. | Label verified. |
| Q025 | `[55, 80]` | Page 55 gives detailed DIAS minimum contents; page 80 gives the broader definition covering DIRM results, impact assessment, initial assurance selection, and tailoring. | Retain both pages; label expansion justified. |
| Q026 | `[84]` | The glossary defines multi-factor authentication as requiring more than one distinct authentication-factor type. | Label verified. |
| Q027 | `[87]` | The glossary defines PIA as analysing collection, use, sharing, and maintenance of personal information and identifying/mitigating privacy risks. | Label verified. |
| Q028 | `[90]` | The glossary defines a salt as a non-secret cryptographic-process value used to help prevent computations from one instance being reused by an attacker. | Label verified. |
| Q029 | `[93]` | The glossary says TLS provides confidentiality and certificate-based authentication of the receiving server and originating client endpoints. | Label verified. |
| Q030 | `[96]` | The change log explicitly says email was removed as a valid channel for out-of-band authenticators. | Label verified. |

## Outcome

- **30/30 question rows reviewed.**
- All labelled target pages directly support their corresponding query.
- Q001 remains `[1, 2]` because both physical pages contain the publication identifier/title.
- Q025 remains `[55, 80]` because page 55 supplies detailed DIAS requirements while page 80 supplies the broad definition.
- No further page-label amendments were supported by this review.
- The fixture's annotation notes now summarize what was checked on each rendered source page.

The previous pinned retrieval benchmark reported **25/30 hits (83.33% Recall@5)** with five misses categorised as `embedding`. This is page-level retrieval quality only, not answer-generation quality. Since this audit changed annotation notes only, not query text or relevant-page labels, the expected Recall@5 remains unchanged. Rerun the validator and pinned benchmark after pulling this commit so the report's dataset hash corresponds to the updated fixture bytes.
