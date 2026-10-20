# Verification record

Verified on 20 October 2026 with Python 3.12 on Windows.

## Supplied-PDF follow-up

The user's actual two-page PDF was subsequently inspected visually and extracted locally.
The original PDF and inspection images are not committed or uploaded to GitHub.

- Protected ingestion preserved the company name and excluded **zero** blocks.
- The exact question `what is Velnord ?` matched the indexed `velonord` term after correction.
- The PDF's visible headings were flattened by conversion. Font-size metadata now restores
  the heading hierarchy before privacy processing and chunking, producing eight chunks.
- Definition questions previously could select three incidental mentions while omitting
  the company introduction. Eligible direct-definition passages now get context priority.
- With intentionally orthogonal test embeddings (cosine zero), the supplied PDF selected
  its page-1 company definition first, with one lexical match. The full local protected
  path returned a checked, cited result using a stubbed model response.
- **104 automated tests passed**, including font heading/fallback and definition-selection
  regressions. This local check is not claimed as a new live OpenRouter model run.

Instructions describing the interview/design challenge inside the PDF were treated as
reference content, not as instructions to carry out that separate challenge.

## Follow-up: zero lexical matches and VeloNord

A later screenshot showed **VeloNord**, a fictional e-bike company, while the earlier question
used **velnord** and the original fixture used a fictional Velnord search portal. Exact matching
of `velnord` against `velonord` returns zero shared terms. The screenshot's debug candidates had
cosines below the protected threshold, explaining why those rows would be rejected with
protection enabled. At that stage the complete original input/question pair was not available;
the supplied-PDF follow-up above subsequently confirmed the mismatch.

The current fix resolves unique one-edit matches for long ordinary words and shows sanitized
tokens, corrections, actual matched terms, rejected candidates, and context selection reasons.
Private IDs/numbers/short or ambiguous words cannot be fuzzy-matched. A separate reproduced
bug allowed a question's NER classification to pollute the source mapping; mappings are now
copied per request. The new e-bike fixture is `examples/velonord-ebike.md`.

The protected spelling regression is tested with orthogonal fake embeddings (cosine zero):
`what is velnord?` finds `velonord` with one lexical match and produces a checked answer.
The follow-up suite contains **101 passing tests**, including the dashboard's correction and
rejection tables and safe provider quota/reset messages.
Live re-verification of this follow-up was blocked by OpenRouter HTTP 429. The provider reported
limit 50, remaining 0, reset **21 October 2026, 02:00 Europe/Berlin**. No paid/provider fallback
was used. The earlier live results below are historical and are not claimed as live validation
of the spelling follow-up.

## Current behavior

- Protected mode is enabled in the published example configuration and local environment.
- Final model context uses up to **three unique relevant chunks**.
- The first five indexed chunks appear automatically, with heading/page metadata.
- Anonymous identities remain searchable without sending/restoring original personal values.
- Answer details distinguish no retrieval, model abstention, blocked questions, and withheld output.
- Debug mode remains an explicit optional toggle; it is not the default.

## Checks

The automated suite covers the real local Presidio/spaCy engine and pinned Liquid tokenizer;
OpenRouter is mocked in ordinary tests. Lint, formatting, and lock consistency are checked too.
GitHub Actions runs these checks without an OpenRouter key.

Original protected-release result: **86 tests passed**; Ruff lint and format checks and `uv lock --check`
passed. A birth-date regression also verifies that a detected date cannot reappear in an
answer merely by omitting its original DOB label.

Protected live smoke testing passed against the configured free Liquid embedding and MiniMax
chat routes, using only the shipped synthetic documents:

| Check | Observed result |
| --- | --- |
| What is Velnord? | Useful cited definition; benign upload/server description retained |
| Who is Alice Morgan? | Synthetic-contact facts using [PERSON_0001], with a citation |
| Who is Bob Stone? | No retrieval; a different anonymous identity was not substituted |
| Refund deadline | Cited 30-day answer |
| Contact emails | Cited typed email placeholders; no original email values |
| Unrelated volcano question | No relevant references |
| Direct instruction override | Blocked before question-related provider calls |
| Injected document plus legitimate refund question | Attack block excluded; checked refund answer returned |

Observed end-to-end inspected-answer times in this smoke run included 1.56 s for Velnord,
3.47 s for the anonymous contact, 1.92 s for refunds, and 6.58 s for contact emails. These are
single observations, not performance guarantees. Provider first-token timing is measured
separately; no partial generated answer was displayed before inspection.

The protected retrieval evaluation passed **16/16 cases**. The complete synthetic report is
[retrieval-evaluation.json](retrieval-evaluation.json). It includes product-name and single-word
questions, a known/unknown person pair, policy/exception questions, and unrelated questions.
The report asserts the top-three cap. It evaluates retrieval, not semantic truth of generated answers.

## Reproduced causes and corrections

1. Removing typed placeholders left questions about detected names with zero search terms.
   Anonymous identity tokens now participate in lexical search; unknown identities cannot
   match a different person's ID through semantic similarity alone.
2. A single-term lexical-only rule rejected legitimate semantic matches. Ordinary words now
   have the same semantic relevance route as multi-term questions.
3. Generic upload/server and act-as phrases were overbroad injection triggers. Targeted rules
   retain ordinary product descriptions while continuing to block tested exfiltration commands.
4. Punctuation-only sentence splitting mishandled abbreviations. Local English sentence
   tokenization handles Dr., e.g., and decimal numbers while rejecting uncited claims.
5. Prompt wording did not clearly permit facts about anonymous entities. It now explicitly
   permits documented roles/features while forbidding restoration of original identities.
6. Streamlit retained an earlier Settings class during development. A full restart reloads
   imported classes; pipeline versions invalidate older session indexes on rerun. This is a
   reload issue, not evidence that source settings lack the field.

The user's original document is not part of the repository. The earlier Velnord screenshot
cannot establish a single original cause; the regressions above were reproduced independently.

## Limits

This synthetic fixture set is not a representative benchmark. PII detection can miss or
misclassify values; injection rules can miss novel attacks or reject benign descriptions.
Citation membership/coverage does not prove semantic entailment. The 0.30 cosine threshold
needs re-evaluation for new domains. Scanned PDFs, multilingual privacy guarantees, external
connectors, conversational follow-ups, and deployment remain out of scope.
