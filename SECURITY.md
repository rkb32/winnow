# Security

Winnow has a public upload site, so security reports are welcome.

## Reporting

Please report privately: on this repository's **Security** tab, choose **Report a vulnerability**. Don't open a public issue for it. Expect a first reply within a week.

## What matters most

- Reading another visitor's photos, reports or past scans, or guessing a guest id.
- Getting around the upload limits (20 photos per scan, the daily and concurrency caps) or the file and filename checks.
- Anything that exposes AWS credentials, bucket names that should be private, or the contents of the `uploads/` and `saved/` prefixes.
- Prompt injection through file names, EXIF text or label files that changes what Claude decides.

## Supported version

The latest commit on `master` and the deployed site.

## Not in scope

Reports that need an attacker to already hold a guest's `#guest=` recovery link (it works like a password), and missing rate limits beyond the caps already described in the README.
