# RuleShift Benchmark

This repository contains the Kaggle notebook and dataset assets for the RuleShift benchmark.

## Main References

If you are working on publishing or maintaining the Kaggle assets, these are the most useful official references.

### Kaggle API and Publishing Docs

- **Kaggle API / CLI repo**: overview of the CLI, installation, authentication, and commands such as `datasets` and `kernels`. ([GitHub][1])
- **Dataset Metadata**: reference for `dataset-metadata.json`, used by `kaggle datasets create` and `kaggle datasets version`. ([GitHub][2])
- **Kernel Metadata**: reference for `kernel-metadata.json`, used by `kaggle kernels push`, including `dataset_sources`. ([GitHub][3])
- **kaggle-api docs index**: entry point for the documentation files in the official repository. ([GitHub][4])

### Kaggle Benchmarks References

- **Community Benchmarks announcement**: official overview of the product and benchmark workflow on Kaggle. ([blog.google][5])
- **`kaggle-benchmarks` on PyPI**: overview of the SDK and its expected behavior inside Kaggle notebooks. ([PyPI][6])

## Recommended Reading Order

If you are new to this workflow, read in this order:

1. **Kaggle API / CLI repo**
2. **Dataset Metadata**
3. **Kernel Metadata**
4. **Community Benchmarks announcement**
5. **`kaggle-benchmarks` on PyPI**

## Notes

- The old dataset/kernel wiki pages indicate that some content was moved into the `docs/` files in the official repository. ([GitHub][7])
- If you only need one entry point for the Kaggle API docs, start with the **kaggle-api docs index**.

[1]: https://github.com/Kaggle/kaggle-api?utm_source=chatgpt.com "GitHub - Kaggle/kaggle-cli: Official Kaggle CLI"
[2]: https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata?utm_source=chatgpt.com "Dataset Metadata · Kaggle/kaggle-api Wiki · GitHub"
[3]: https://github.com/Kaggle/kaggle-api/wiki/Kernel-Metadata?utm_source=chatgpt.com "Kernel Metadata · Kaggle/kaggle-api Wiki · GitHub"
[4]: https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md?utm_source=chatgpt.com "kaggle-api/docs/README.md at main · Kaggle/kaggle-api · GitHub"
[5]: https://blog.google/innovation-and-ai/technology/developers-tools/kaggle-community-benchmarks/?utm_source=chatgpt.com "Community Benchmarks: Evaluating modern AI on Kaggle"
[6]: https://pypi.org/project/kaggle-benchmarks/?utm_source=chatgpt.com "kaggle_benchmarks · PyPI"
[7]: https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata "Dataset Metadata · Kaggle/kaggle-cli Wiki · GitHub"
