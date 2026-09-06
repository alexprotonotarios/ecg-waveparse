# Registry configuration prepared for the release owner

No package has been published. On 5 September 2026, both registry metadata
endpoints returned HTTP 404 for `ecg-waveparse`. This does not reserve either
name or establish account ownership.

## Maintainer accounts

Both personal accounts were verified in their authenticated settings on
6 September 2026, under the username `alexprotonotarios`:

- npm: two-factor authentication enabled for authorization and publishing,
  with one registered security key.
- PyPI: email verified, authenticator application enabled, and recovery-code
  verification completed.

Account creation is complete. Package ownership is established by the first
publication, not by account registration. Retain passwords and recovery codes
privately; do not put them or API tokens in source files or messages.

## Publication configuration

| Setting | Intended value |
| --- | --- |
| npm package / Python distribution | `ecg-waveparse` |
| Python import | `ecg_waveparse` |
| GitHub owner | `alexprotonotarios` |
| Clean repository | `ecg-waveparse` |
| Publication workflow | `publish-waveparse.yml` |
| Protected publication environment | `release` |
| Initial prepared version | `0.1.0` |

The package repository metadata points to the existing private clean repository.
Registry ownership, authentication and public visibility remain separate from
local package verification. The existing data-bearing `ecg_digitizer` repository must stay
private. Do not copy its git history into the release repository.

The publication workflow is prepared with `verify` as its default target.
It downloads the two native artifacts from a successful, manually dispatched
`waveparse-packages.yml` run on this repository's `main` branch. It checks the
full source commit, both successful native jobs, source-audit and inference
evidence, and four independently reviewed Ubuntu archive SHA-256 values. npm
and wheel bytes must match on macOS; both source archives must contain the same
regular files across hosts. No package is rebuilt or executed in this workflow.

The npm archive, wheel/sdist and corresponding source are separated into
`npm/`, `python/` and `source/` directories. The verification receipt has
`published: false`; a passing verification run does not mean a registry release
occurred. The source archive is not uploaded to PyPI as a Python distribution.

Registry targets remain disabled unless the repository variable
`WAVEPARSE_PUBLICATION_ENABLED` is explicitly `true`, and they require the clean
repository to be public. Before enabling it, complete the release decision in
LICENSING.md, configure the registry publishers and the `release` environment,
and review the exact successful run and artifact identities. Keep the variable
unset while the model enquiries are outstanding. This is the current release
process, not a legal determination that publishing the wrapper alone is prohibited.

Set the GitHub `release` environment to allow only `main` and require the release
owner's approval where the repository plan supports required reviewers. Protect
the release workflow and main branch from unreviewed changes. The build/test job
has no registry credentials or OIDC permission; only the separate publication
jobs request short-lived publishing credentials.

For PyPI, sign into the intended maintainer account and add a pending GitHub
publisher using the table above. It creates the project only when an authorised
publication occurs; it does not reserve the name. TestPyPI is a separate registry
and needs separate configuration.

For npm, the first publication uses the maintainer's interactive CLI login and
security key. After the verification-only workflow succeeds, download its
`waveparse-verified-release` artifact, recheck the receipt hashes and run
`npm publish npm/ecg-waveparse-0.1.0.tgz --ignore-scripts --access public --tag next`
from that artifact directory. This command is for the approved release step,
not account setup. The package's preparation-only `prepublishOnly` guard stays
in the archive; `--ignore-scripts` deliberately prevents package scripts from
executing when uploading these already verified bytes.

Then configure npm's trusted publisher with the exact repository, workflow
filename and environment above, explicitly allowing `npm publish`. Future
versions can use the workflow's npm target. Staged publication is available for
existing packages, but cannot bootstrap a brand-new package. Do not publish a
placeholder merely to claim the name.

Use GitHub-hosted runners and short-lived OIDC for subsequent releases. npm
requires CLI 11.5.1 or newer and Node 22.14.0 or newer; the prepared runtime uses
Node 22.22.2. The publication job pins npm CLI 11.15.0 separately from the inference runtime.
Automatic npm provenance requires a public source repository as well as a public
package, so approve clean-source visibility before the actual public release.

After each registry upload, verify that the expected version and owner are
visible, download the registry archives and compare them with the reviewed
SHA-256 values, and run fresh Python and JS/TS consumer installations. Record
each registry's confirmation separately: publication to one registry can succeed
while the other fails. A failed second upload must not trigger a replacement
build or a version overwrite. Application upgrades remain separate from package
publication.

## Current remaining steps

1. Record the model-licence answers and the release owner's final licence/scope decision.
2. Configure PyPI's pending publisher, GitHub's `release` environment and the
   initial npm CLI session. No long-lived publishing token is needed.
3. Run supported-platform CI on the final release source. Run the publication
   workflow with `target: verify`, its full source commit and four approved
   Ubuntu archive hashes. Earlier CI archives can exercise this verification
   path, but must not be confused with the final release candidate.
4. Make the clean source repository public, enable the release variable for the
   approved upload, publish to npm and PyPI, then verify registry downloads and
   fresh installations. Configure npm's trusted publisher after its first upload.

References checked for this preparation:

- [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/)
- [npm staged publishing prerequisites](https://docs.npmjs.com/staged-publishing/)
- [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
