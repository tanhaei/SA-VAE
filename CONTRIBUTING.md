# Contributing

1. Never commit patient data, credentials, linkage keys, or model artifacts that
   can disclose protected information.
2. Add or update a unit test for every behavior change.
3. Run `make check` before opening a pull request.
4. Preserve patient-disjoint splitting and field-specific target masking.
5. Do not label synthetic results as clinical reproduction.
6. Document any new baseline's exact version, hyperparameters, input window, and
   missingness assumptions.

