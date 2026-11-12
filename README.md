# Flag algebra certificate for the inducibility of K_{4,2,1^9}

This repository contains the code and flag algebra certificate accompanying the study of inducibility and perfect stability for complete partite graphs. In particular, it certifies the inducibility of $K_{4,2,1^9}$:

$$
 \lambda_{K_{4,2,1^9},\max}=\frac{118216098}{1220703125}.
$$

The calculation uses four cluster-graph types and 176 partitions of 15. The resulting certificate is checked using exact integer and rational arithmetic.

## Files

| File | Purpose |
| --- | --- |
| `flag_model.py` | Enumerates partitions, flags, and flag-product matrices. |
| `face.py` | Constructs the reduced search space. |
| `sdp_core.py` | Provides numerical matrix utilities. |
| `search_sdp.py` | Searches numerically for an SDP certificate with the target bound fixed. |
| `recover_certificate.py` | Converts the numerical solution into a rational certificate. |
| `verify_certificate.py` | Recomputes and verifies the certificate using exact arithmetic. |
| `verify_independent.py` | Independently recomputes the flag products and verifies the certificate. |
| `check_local.py` | Checks the perturbation and finite-construction calculations. |
| `regression_tests.py` | Runs small-case and corrupted-certificate tests. |
| `run_all.py` | Runs the complete computation and verification pipeline. |
| `reference_run/` | Contains a reference certificate, numerical outputs, logs, and verification reports. |

## Reproduce the computation

Install Python and the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the complete search, rational recovery, and verification in a **new or empty** output directory:

```bash
python run_all.py --out results
```

A successful run ends with `COMPLETE_EXACT_VERIFIED`. The new certificate is written to `results/sharp_certificate.json.gz`.

## Verify the supplied certificate

To check the certificate without repeating the numerical SDP search, run:

```bash
python verify_certificate.py --certificate reference_run/sharp_certificate.json.gz --out verified_check
python verify_independent.py reference_run/sharp_certificate.json.gz independent_check
```

These verification scripts use the Python standard library and exact arithmetic. They check the positive semidefiniteness conditions, all 176 coefficient identities, and the nonnegativity of the residuals. The two implementations independently reconstruct the 704 flag-product matrices.

The numerical search is a means of finding a certificate; the mathematical upper bound rests on the exact verification.
