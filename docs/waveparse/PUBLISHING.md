# Registry configuration prepared for the release owner

No package has been published. On 5 September 2026, both registry metadata
endpoints returned HTTP 404 for `ecg-waveparse`. This does not reserve either
name or establish account ownership.

| Setting | Intended value |
| --- | --- |
| npm package / Python distribution | `ecg-waveparse` |
| Python import | `ecg_waveparse` |
| GitHub owner | `alexprotonotarios` |
| Clean repository | `ecg-waveparse` |
| Future publication workflow | `publish-waveparse.yml` |
| Protected publication environment | `release` |
| Initial prepared version | `0.1.0` |

The package repository metadata points to the intended clean repository. Its
creation/visibility and registry authentication are separate from local package
verification. The existing data-bearing `ecg_digitizer` repository must stay
private. Do not copy its git history into the release repository.

Before configuring publication, complete LICENSING.md, inspect the source audit,
and obtain a passing supported-platform workflow for the exact engine payload.
Keep `prepublishOnly` disabled until the concrete release is approved; the current
package workflow has no registry credentials, OIDC token permission or publish
step.

For PyPI, sign into the intended maintainer account and add a pending GitHub
publisher using the table above. It creates the project only when an authorised
publication occurs; it does not reserve the name. TestPyPI is a separate registry
and needs separate configuration.

For npm, verify the intended maintainer identity and first-release route before
creating the package. The documented trusted-publisher setup is in an existing
package's settings. Configure the exact repository, workflow filename and
environment after establishing package ownership. Do not use a placeholder
publication merely to claim the name.

Use GitHub-hosted runners and short-lived OIDC for subsequent releases. npm
requires CLI 11.5.1 or newer and Node 22.14.0 or newer; the prepared runtime uses
Node 22.22.2. Pin the publishing CLI separately from the inference runtime.
Automatic npm provenance requires a public source repository as well as a public
package, so approve clean-source visibility before the actual public release.

The future publish workflow should take an approved successful package run and
source revision, verify the saved SHA-256 values, and publish those artifacts.
It should not rebuild an untested payload during publication. Verify registry
downloads against the approved identities, then upgrade Neo's dependency in a
separate reviewed change. Publishing a package does not update deployed workers.

References checked for this preparation:

- [npm trusted publishing](https://docs.npmjs.com/trusted-publishers/)
- [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
