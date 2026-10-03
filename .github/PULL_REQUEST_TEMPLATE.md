## What this changes

<!-- One change per pull request. -->

## What you measured

<!-- Numbers, on which data, and how to rerun. Report misses and false alarms next to the hits. -->

## Checklist

- [ ] `docker build --target test winnow/` passes
- [ ] A bug fix comes with a test that fails without it
- [ ] If a threshold or check changed: the Imagenette audit was rerun, and the real leaks still flagged are reported
- [ ] If a setting was chosen while looking at one dataset's answer key, the result is labeled in-sample
- [ ] No photos from a dataset, and no keys, tokens or private ARNs
- [ ] A new module in `winnow/` is added to the `COPY` line in `winnow/Dockerfile`
