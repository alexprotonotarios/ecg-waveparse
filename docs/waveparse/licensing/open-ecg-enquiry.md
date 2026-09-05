# Open-ECG-Digitizer model licence enquiry

Status: approved by the release owner and posted as
[issue 47](https://github.com/Ahus-AIM/Open-ECG-Digitizer/issues/47) on
5 September 2026. Reloaded and verified against the approved text; awaiting a
maintainer answer. No email was sent.

Destination: a public issue in
[Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer/issues).
The pinned README also lists `elias.stenhede@ahus.no` for questions, if email is
preferred. Posting an issue and sending an email are alternatives, not two requests.

Subject: Clarify CC BY-SA 4.0 coverage of the two released model weights

## Proposed message

Hello,

Thank you for releasing Open-ECG-Digitizer. I am preparing WaveParse, an
installable Python and JavaScript interface to a local ECG digitisation runtime.
The runtime pins your repository at
`97a15087d4abcda843da8c58ee74b1d8f47e6f9a`.

I found the repository's CC BY-SA 4.0 licence and the response to issue #43.
Could you confirm that the same licence covers these released weight files?

| File | SHA-256 |
| --- | --- |
| `weights/unet_weights_07072025.pt` | `17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b` |
| `weights/lead_name_unet_weights_07072025.pt` | `840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2` |

Users explicitly download your unmodified engine and weights from upstream
during setup; our language packages do not bundle or mirror them. We retain
your licence, attribution and research citation. Adapted layout definitions and
Python subclass extensions are marked CC BY-SA 4.0, alongside separately
authored interface code under MIT.

Are commercial local inference and redistribution of these weights covered by
CC BY-SA 4.0, or do any separate model terms, additional notices or restrictions
carried through from the training data apply? Please also flag any attribution
or licence-boundary concern with the arrangement described above.

A model-specific licence statement or model card would be helpful for downstream
users. Thank you,

Alexandros Protonotarios

## Preparation evidence

- [Pinned README](https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/README.md)
- [Pinned licence](https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/97a15087d4abcda843da8c58ee74b1d8f47e6f9a/LICENSE)
- [Maintainer reply to issue 43](https://github.com/Ahus-AIM/Open-ECG-Digitizer/issues/43#issuecomment-4620497847)

The existing reply confirms that a repository licence was added; it does not
explicitly discuss the two model files. No model-specific answer was identified
in the licence-related issues inspected on 5 September 2026.
