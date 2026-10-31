# Tasks - image-cuda128-diet

Strict TDD per `docs/TESTING.md`: behaviour written as a failing test first.

## 1. Diet script (RED -> GREEN)

- [x] 1.1 RED: `tests/test_image_diet.py` against a fake `site-packages` tree
  (9 tests: removals, kept wheels, triton, torch headers/tests, dist-info, dry
  run, missing kept wheel, JSON report, bad path) -> 9 errors, script absent
- [x] 1.2 GREEN: `scripts/image_diet.py` (`plan`, `apply`, `--apply`, `--json`,
  `--site-packages`, default path from `sysconfig`) -> 9 tests OK
- [x] 1.3 RED/GREEN: purge of the build toolchain, `ldconfig`, no `autoremove`
- [x] 1.4 Keep the two wheels the base lacks and fail closed if absent (covered
  by 1.1: `test_missing_kept_wheel_fails_loudly`)

## 2. Build configuration

- [x] 2.1 RED: five tests in `tests/test_cog_yaml.py` (base CUDA 12.8, diet
  applied once, no autoremove, `LD_LIBRARY_PATH`, requirements contract)
- [x] 2.2 GREEN: `cog.yaml` (cuda 12.8, diet step, purge, `ldconfig`,
  `LD_LIBRARY_PATH`)
- [x] 2.3 `make -f Makefile.harness check` green (600 tests)

## 3. Documentation

- [x] 3.1 `docs/IMAGE_SIZE.md`: layer breakdown, `readelf` evidence, decisions,
  measured value after the rebuild
- [x] 3.2 `PLANS.md`: entry for this change

## 4. Verification before done

- [x] 4.1 `make -f Makefile.harness check` green (600 tests)
- [ ] 4.2 `make -f Makefile.harness ci` green
- [ ] 4.3 Image rebuilt by the publish workflow; size measured from the registry
  and written in `docs/IMAGE_SIZE.md` (expected 9.40 GB -> about 6.2 GB)
- [ ] 4.4 GPU smoke on the canary (transcribe + align + diarize on a real clip),
  transcript compared with the previous image
- [ ] 4.5 Promotion to production (operator decision)
