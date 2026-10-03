# Contributing to Winnow

Fixes, new checks and outside tests are all welcome. Contributions are accepted under the project's [Apache 2.0 license](LICENSE).

## Before you start

For anything bigger than a fix, open an issue first so we can agree on the approach. A good small contribution is [`binder_eval/`](binder_eval/README.md): a script, its output, and a README section that says what was measured and how to rerun it.

## Run the tests

```bash
docker build --target test winnow/
```

This runs the whole suite inside the same image that ships (same OpenCV 5 build, same embedding model). A pull request needs it to pass.

- A bug fix should come with a test that fails without the fix.
- A new module in `winnow/` must also be added to the `COPY` line in `winnow/Dockerfile`, or it will be missing from the image.
- Dependencies are pinned in `winnow/requirements.txt`. Pin any you add.

## Changing a threshold or a check

Winnow's thresholds were calibrated on Imagenette and hand-verified (`imagenette_exp/`). If a change affects what gets flagged:

- Rerun the audit and report how many of the 76 real leaks stay flagged and how many false alarms change.
- Say which data you measured on. If you chose a setting while looking at one dataset's answer key, label that result in-sample.
- Describe what you did not test, such as a kind of data the check has not seen.

## Datasets and photos

Do not add image datasets to a pull request. The photos in a benchmark usually belong to someone else. File names, pair lists, answer keys and numbers are fine.

## Never commit

AWS keys, tokens, account-specific ARNs you don't want public, or anything from a private bucket. Secret scanning is on for this repository, but check before you push.

## Pull requests

- One change per pull request, with a description of what it does and what you measured.
- Keep results honest: report misses and false alarms next to the hits.
