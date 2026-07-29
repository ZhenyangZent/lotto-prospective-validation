# Gitleaks false-positive review

- Finding rule: `generic-api-key`
- Finding commit: `869d4dc4cf26d98075d81d0b7dacca9ff38594ce`
- Finding location: `prospective_validation_v2/remote_verification.json`, line 5
- Reviewed field: `github_api_commit_sha`
- Reviewed value classification: public Git commit object identifier (40 lowercase hexadecimal characters)

The reviewed value is identical to `prediction_commit`, is the final path component of
`github_commit_url`, resolves locally as a Git `commit` object with `git cat-file`, and
is returned unchanged by the GitHub commits API. It has no GitHub PAT, OAuth token,
API-key, or credential prefix and is public evidence rather than authentication data.

The repository allowlist is deliberately conjunctive. It applies only when the finding
uses the `generic-api-key` rule, the path is exactly
`prospective_validation_v2/remote_verification.json`, and the complete line contains
only the `github_api_commit_sha` field with one 40-character lowercase hexadecimal Git
OID. It does not allow another line in that file, another field, another path, another
rule, or a non-40-character value. The evidence JSON and existing Git history were not
modified.

Verification performed on 2026-07-29:

- `git cat-file -t <oid>` returned `commit`.
- `git rev-parse <oid>^{commit}` returned the reviewed OID.
- GitHub REST `repos/ZhenyangZent/lotto-prospective-validation/commits/<oid>` returned
  the reviewed OID.
