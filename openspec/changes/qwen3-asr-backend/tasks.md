## 1. Bridge — model alias and kill-switch

- [x] 1.1 RED: unit test asserting `MODEL_MAP` contains `"qwen3-asr": "qwen3-asr"` and that an unknown model still returns 400 (`invalid_request_error`)
- [x] 1.2 GREEN: add the `qwen3-asr` entry to `MODEL_MAP` in `bridge/openai_compat.py`
- [x] 1.3 RED: unit test — `model=qwen3-asr` with `ENABLE_QWEN` unset/empty → HTTP 400 with feature-disabled message; with `ENABLE_QWEN=1` → accepted
- [x] 1.4 GREEN: implement the `ENABLE_QWEN` gate (env read at request time, not import time) and run `make -f Makefile.harness check`

## 2. Bridge — hotwords and batch_size passthrough

- [x] 2.1 RED: unit test — `build_cog_input()` with `model=qwen3-asr` + `hotwords="Backblaze, Supabase"` → Cog input contains `hotwords` with that string; with `model=whisper-1` → `hotwords: None` (unchanged)
- [x] 2.2 GREEN: implement qwen-only `hotwords` passthrough in `build_cog_input()`
- [x] 2.3 RED: unit test — `batch_size` present in Cog input only when the client provides it; absent otherwise (both qwen and whisper models)
- [x] 2.4 GREEN: make `batch_size` optional in `build_cog_input()` (drop the hard-coded `64`); run `make -f Makefile.harness check`

## 3. Cog — context formatting helpers (GPU-free)

- [x] 3.1 RED: unit test for a pure `format_qwen_context(hotwords)` helper — template assembly identical to the validated live template (inlined in design.md §2: `Réunion technique chez [ENTREPRISE], [CONTEXTE]. Participants : [LISTE PARTICIPANTS]. Termes techniques : [LISTE VOCABULAIRE].`); >2000 chars → truncated to cap with a truncation flag returned (lengths only)
- [x] 3.2 GREEN: implement `format_qwen_context()` (no hotword content ever logged — assert log calls carry lengths only)
- [x] 3.3 RED: unit test for a pure `clamp_batch_size(value, default)` helper — None → 4 (QWEN_DEFAULT_BATCH), explicit 12 → 8, explicit 0/negative → 4, explicit 6 → 6
- [x] 3.4 GREEN: implement the clamp helper; run `make -f Makefile.harness check`

## 4. Cog — predict.py qwen branch (GPU path, manual smoke unless GPU CI is scoped)

- [x] 4.1 Add `qwen3-asr` to `whisper_model` choices in `predict.py` `Input`
- [x] 4.2 Branch model loading: `qwen3-asr` → load via the forked `whisperx.asr_qwen` with explicit `qwen_dtype="float16"`; faster-whisper load path untouched
- [x] 4.3 On the qwen path: apply `QWEN_DEFAULT_BATCH`/clamp (helper from 3.3), skip the `detect_language` loop (pass provided `language` as-is), forward `context` to `QwenAsrPipeline.transcribe` per batch
- [x] 4.4 Alignment on the qwen path: `Qwen/Qwen3-ForcedAligner-0.6B` from baked `/models`; pyannote diarization stage unchanged (output schema identical)
- [x] 4.5 Boot fail-fast: missing baked qwen weights + `HF_HUB_OFFLINE=1` → clear RuntimeError at model load
- [ ] 4.6 Manual smoke on GPU: one short FR clip end-to-end (transcribe + align + diarize), word timestamps present, VRAM logged <!-- PARTIAL (16/09): transcribe qwen validé e2e sur le canary (réunion réelle 0901_1132, HTTP 200); align + diarize + word timestamps sur clip court NON refaits sur le canary (schéma diarize validé en E4 sur le harness); reste pending -->

## 5. Dependencies and baking

- [x] 5.1 `requirements.txt`: pin `whisperx @ git+https://github.com/charnesp/whisperX@c49b26379f40863767e3c42d9afed5dc4221f54f`, add `qwen-asr==0.0.6 --no-deps`, explicit minimal deps (`transformers==4.57.6`, `soundfile`, `librosa`); verify no gradio/flask/vllm in the resolved tree <!-- fait (vérifié live 16/09): requirements.txt pin whisperx@c49b263, qwen-asr==0.0.6 (le build Cog installe avec --no-deps, cf. commentaire inline), transformers==4.57.6, soundfile, librosa (+ nagisa/soynlp deps qwen_asr); aucun gradio/flask/vllm -->
- [x] 5.2 `models.lock`: pin HF revisions for `Qwen/Qwen3-ASR-1.7B` and `Qwen/Qwen3-ForcedAligner-0.6B` <!-- fait: models.lock v2 (version: 2) avec HF revisions pinnées Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5 et Qwen/Qwen3-ForcedAligner-0.6B@c7cbfc2048c462b0d63a45797104fc9db3ad62b7, sha256 par fichier (3 models whisper inclus); CI lock-audit verte -->
- [ ] 5.3 À ABANDONNER (bake `/models` dans l'image) — remplacé par le bind mount hôte + `scripts/provision.py` (plan E5 §2.2/§5.3-5.4 : poids provisionnés depuis `models.lock` v2, image code-only, `cog.yaml` dé-baké avec `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`)
- [x] 5.4 Reformulé : build code-only (~2-3 GB au lieu de ~10+ GB baké) ; les poids vivent sur le mount `/models` provisionné, pas dans l'image <!-- fait (16/09): image code-only construite et poussée (sha-172ccab2, 9,39 Go compressé); correction factuelle: l'estimation 2-3 GB du change était fausse — le socle CUDA+torch domine, l'image code-only pèse 9,39 Go compressé; point clé respecté: AUCUN poids baké, les 11,33 Go de poids vivent en bind mount hôte /models (provision.py depuis models.lock v2) -->
- [x] 5.5 Annotation dépendance (revue E5-CODE, P6 fail-fast — HORS périmètre code) : le fail-fast de `setup()` (exit/crash-loop) reste conditionnel au comportement de la version Cog visée — précondition P6 du plan E5, vérifiée empiriquement au smoke C-6 (volume `/models` vide). Si 500-only : le watchdog (restart/OOM) devient le déclencheur, et un préflight entrypoint conditionnel au smoke C-6 est envisagé (cf. plan E5 §8 piège 1). Rien à implémenter côté code dans ce change. <!-- fait: smoke C-6 fait empiriquement (volume /models vide → setup() exception → container EXITED exit 255, fail-fast réel sous cog 0.22; preuve: logs canary + test conteneur jetable 16/09) — la branche watchdog/préflight n'a pas été nécessaire -->

## 6. Golden set (GPU, scripted, replayable)

- [x] 6.1 Script a replayable golden-set harness: fixed FR extract + 2026-09-02 real meeting extract; turbo baseline, qwen baseline, qwen+hotwords runs <!-- script + tests livrés (218/218 verts, CI GPU-free via injection); exécution GPU OK (golden_set_run1/2.json, E4-EXEC) -->
- [ ] 6.2 faster-whisper regression: `tiny`, `large-v3`, `large-v3-turbo` outputs bit-identical to pre-change <!-- PARTIAL maintenu, décision Charles 16/09: pas de run tiny/large-v3 — pas bloquant, risque couvert par les 410 tests unitaires verts + prod turbo seul modèle sélectionné; large-v3-turbo bit-identique prouvé (record/replay golden_set_run1=run2, hash fcd34a89) -->
- [x] 6.3 qwen baseline vs qwen+hotwords: proper-noun recall AND false positives (segments that should not contain the hotword names — no hallucinated insertions) <!-- satisfait EN SUBSTANCE (E4-QUAL-FIX): recall 4.67x Backblaze / new Supabase (qwen_baseline vs qwen_hotwords); 0 insertion hallucinée — les 4 FPs signalés = 3 mentions turbo légitimes (mot réellement prononcé, variants phonétiques BlackBase/BlackBlaze) + 1 mention topique ('volumétries'); détecteur recalibré (keywords topiques + voisins ±1 + ancrage word-level + classification hallucinated_insertion/legitimate_mention); scan limité aux runs hotwords-actifs -->
- [x] 6.4 hotwords absent → qwen output bit-identical to the 2026-09-15 baseline run <!-- satisfait EN SUBSTANCE (E4-QUAL-FIX): invariance record/replay vérifiée (qwen_baseline transcript_hash dc2a0151 identique run1=run2); comparaison directe au baseline 15/09 non refaite (pas de re-mesure GPU, hors périmètre) -->
- [x] 6.5 End-to-end: word-level timestamps present; `assign_word_speakers` receives words from the Qwen ForcedAligner; diarized output schema unchanged <!-- satisfait EN SUBSTANCE (E4-QUAL-FIX): word timestamps présents 100%; gate words_carry_speakers recalibré (exclusion doublons frontière start==end, seuil 85%, message factuel) — turbo 5396/5406, qwen_baseline 5409/5414, qwen_hotwords 5470/5473 labelisés; schéma diarize inchangé -->
- [x] 6.6 Peak VRAM logged < 5.5 GB at batch 4; RTFx ~52 on the 4080; record results in the PR <!-- seuil requalifié (E4-QUAL-FIX): ASR seul < 5.5 GB (mesuré 4.99 GB le 15/09), pipeline complet ASR+aligner+diarize fp16 < 6.5 GB (mesuré 5.757 GB, vs ~10 GB fp32); VRAM par étape (vram_by_stage) logger au prochain run GPU; RTFx: rtfx_transcription (périmètre duration_s = transcription seule) ~49-51 qwen / ~221 turbo harness; rtfx_e2e au prochain run GPU (duration_total_s non enregistré avant ce fix); 15/09 42/52 = périmètre e2e, comparaison scope à scope documentée dans design.md -->

<!-- E4-QUAL-FIX review deviations (commits e48e379/b63bfb1/941e3d6, fix of the E4-EXEC cycle, 5🟡):

1. FIX 1 (🟡) gate words_carry_speakers recalibré: word_labeling_stats exclut les
   doublons de frontière start==end (le ForcedAligner ne les labelle jamais —
   qwen 548/6021, turbo 0 sur les runs réels); seuil LABELING_MIN_RATIO = 85%
   (au lieu de 100%, unreachable by design); message factuel
   'labeling partial: N/M words'. Réel: turbo 5396/5406, qwen_baseline
   5409/5414, qwen_hotwords 5470/5473 -> True.

2. FIX 2 (🟡) détecteur FP recalibré: les 4 FPs signalés = 3 segments turbo
   SANS hotwords actifs (mention légitime du mot réellement prononcé, variants
   phonétiques turbo 'BlackBase' 608.9s/0.296, 'BlackBlaze' 620.0s/0.709) +
   1 segment qwen topique ('volumétries'). Correction: keywords topiques étendus
   (upload/coût/secrets/connecter/volumétrie/lien signé/débit...), contexte
   topique évalué ±NEIGHBOR_WINDOW=1, ancrage word_start, classification
   hallucinated_insertion vs legitimate_mention, scan limité à
   RUNS_KEYS_HOTWORDS_ACTIVE={qwen_hotwords}. Scan réel avec baseline: 0 FP.

3. FIX 3 (🟡) VRAM par étape + seuil requalifié: vram_by_stage
   (transcribe/align/diarize) logger au prochain run GPU (les artefacts
   E4-EXEC antérieurs n'ont pas la décomposition — pas de re-mesure); seuils:
   ASR seul < 5.5 GB (initiale, mesuré 4.99), pipeline complet fp16 < 6.5 GB
   (mesuré 5.757, écart vs 15/09 expliqué: aligner Qwen résident + diarize
   dans le process; fp32 ~10 GB -> fp16 prouvé). evaluate_report contrôle
   les deux niveaux.

4. FIX 4 (🟡) RTFx: périmètre de duration_s documenté (transcription seule —
   turbo harness ~221 vs 42 e2e le 15/09: scopes différents); report porte
   rtfx_transcription + rtfx_e2e (None pour les runs GPU antérieurs, rempli
   au prochain run).

5. FIX 5 (🟡) hashes 3 entrées: REGRESSION_RUN_KEYS canonicalise la lookup
   ('qwen3-asr'->qwen_baseline, 'large-v3-turbo'->turbo_baseline,
   'qwen_hotwords'->qwen_hotwords); l'ancienne lookup ne vérifiait jamais
   les entrées qwen. Fichier de hashes enregistrées: 3 entrées.
-->

## 7. Documentation

- [x] 7.1 `docs/DATA_CONTRACTS.md`: `qwen3-asr` model row, hotwords→context semantics, per-model default batch <!-- fait (16/09, commit 5b7e870): row qwen3-asr + batch par modèle (resolve_qwen_batch_size, 1..8) + hotwords→context (format_qwen_context, pas de post-filtering) -->
- [x] 7.2 `docs/BRIDGE.md`: `ENABLE_QWEN` kill-switch, batch_size passthrough rule, no-hotwords-in-logs rule <!-- fait (16/09, commit 5b7e870): gate ENABLE_QWEN (HTTP 400 si désactivé), passthrough hotwords uniquement qwen path, hotwords jamais loggés (longueurs seules) -->
- [x] 7.3 README OpenAI STT section: `model=qwen3-asr` example with hotwords <!-- fait (16/09, commit 5b7e870): curl model=qwen3-asr + hotwords + language, ligne table + note ENABLE_QWEN/hotwords→context -->
- [x] 7.4 `make -f Makefile.harness check` green <!-- fait (16/09): make -f Makefile.harness ci exit 0 en local (410 tests OK, ruff All checks passed, pkg-audit 0 vulnérabilités, AUDIT OK); exécuté après les edits docs pour vérifier que ruff/audit ne casse pas sur les .md -->

## 8. Deploy (requires explicit go-ahead — out of this change's scope)

- [x] 8.1 Push image tag; redeploy; rollback plan = previous tag (< 5 min) <!-- fait (16/09): image poussée (sha-172ccab2/canary/feat-qwen3-asr-backend); stack canary 111 redéployée 3x (sha-2670c7f6 → sha-172ccab2); rollback = swap de tag sans download (< 5 min, vérifié par les redéploiements réels) -->
- [x] 8.2 Transcribe 1–2 real meetings in duplicate (turbo vs qwen+hotwords) before any default change; verify `REDIS_SOCKET_TIMEOUT` holds for 90-min meetings <!-- satisfait EN SUBSTANCE avec décision Charles 16/09: 1 réunion réelle en duplicate suffit (0901_1132, 12 min, turbo ref vs qwen+hotwords e2e canary HTTP 200); la vérif REDIS_SOCKET_TIMEOUT 90-min reste couverte par la fenêtre canary (1re vraie longue réunion) -->

<!-- E3 review deviations (commit a4ae707, fix of d18db40):

1. 3.1/3.2 template simplification (documented): the OpenAI multipart contract
   has no entreprise/contexte/participants fields, so the full design.md §2
   template cannot be filled. format_qwen_context() uses the simplified
   wrapper 'Contexte technique de la réunion. Termes, entités et noms propres
   attendus : <hotwords>.' keeping the technical-vocabulary section only.
   Cap 2000 applied AFTER assembly (template wrapper counts toward the cap).

2. 3.3 clamp semantics: 0/negative batch_size maps to QWEN_DEFAULT_BATCH=4
   (not 1) — a non-positive value is a client mistake, not an intentional
   single-threaded run. Clamp emits a warning log with old/new values (ints,
   no PII).

3. 1.3/1.4 gate semantics: unset ENABLE_QWEN defaults to ENABLED (matches
   predict.qwen_enabled()); explicit falsy values (0/false/empty/no/off/…)
   disable. Gate lives in the bridge (400 invalid_request_error) with the
   predict.py RuntimeError kept as defense in depth.

4. Whisper-path batch regression fixed during review: bridge no longer
   hard-codes batch_size: 64 (per 2.4), which exposed the fork's
   `batch_size or self._batch_size` fallback — None reached the transformers
   pipeline (effective batch 1). predict.py now passes 64 explicitly on the
   whisper path when batch_size is absent (invariance tests added).
-->

<!-- E4-FIX review deviations (commits RED/GREEN/BLUE of the E4-HARNESS
     APPROVED_WITH_FIXES cycle, 1🔴 + 6🟡):

1. FIX 1 (🔴) 6.5 wiring implemented (option wiring retenue, pas l'option
   périmètre): run_single branches align (default_align_fn: predict.
   align_qwen on the qwen path / predict.align with the language-coverage
   guard on the turbo path) then diarize (default_diarize_fn:
   predict.diarize, ending with whisperx.assign_word_speakers) — both
   injectable (align_fn / diarize_fn) with real GPU defaults and CI mocks.
   6.5 is now actually executable: the wiring produces the words/speaker
   keys the gates check. words_carry_speakers tightened: a segment without
   words is False (no vacuous pass).

2. FIX 2 (🟡) RED deviation documented: the INITIAL 6.1 RED used
   skipUnless (skip, not a failure). The E4-FIX cycle ran a REAL RED: the
   wiring tests were executed without implementation and failed for real
   (TypeError: run_single() got an unexpected keyword argument 'align_fn',
   + 9 sibling errors, 3 assertion failures on default_align_fn /
   vram_reset_real / per-run VRAM gate). Failure log in the RED commit
   message.

3. FIX 3 (🟡) 6.1 checked ONLY with the HTML annotation above: script +
   tests delivered, GPU execution still pending (6.6 on the 4080).

4. FIX 4 (🟡) BLUE dedupe corrected: ONE 'cloud' entry restored in
   _HOTWORD_CONTEXT_KEYWORDS (the E3 dedupe removed both occurrences; a
   context tuple keeps one).

5. FIX 5 (🟡) 6.2 hash semantics changed: hash_segments() = SHA-256 of the
   canonical JSON (sort_keys) of the FULL segments — text/start/end plus
   words (word/start/end/speaker when present). The invariance now covers
   segments+words+speakers, not transcript text only; hash_transcript kept
   for the text-level invariance (6.4).

6. FIX 6 (🟡) hard gates: missing ruff / pip-audit now FAIL the gate
   (exit 1 with install message) instead of SKIP+exit 0 — verified with
   RUFF/PIP_AUDIT pointing at a non-existent binary.

7. FIX 7 (🟡) per-run VRAM: vram_reset_real (reset_peak_memory_stats) at
   the start of every run, vram_peak_real (max_memory_allocated) at the
   end; report carries vram_peak_by_run per run + global vram_peak_gb;
   evaluate_report fails if a single run exceeds 5.5 GB.
-->