# Release validation

Validated on 17 October 2026 using Python 3.9.18 and PyTorch 2.3.0+cu118.

- All packaged Python files parse successfully.
- The nine generative `model.py` files match the original project source after line-ending normalization.
- Eight standalone generative models pass small CPU loss, gradient and sampling checks.
- The Transformer predictor passes forward/backward checks; the full architecture also loads the existing research checkpoint strictly and produces finite predictions.
- PA-FM-CFG passes loss, gradient and sampling checks with the existing frozen research surrogate loaded locally.
- Predictor HPO completes two short trials using temporary random input data and a small test architecture; the selected configuration and saved checkpoint reload consistently. Temporary test data are not part of the repository.
- The generative HPO, inference, predictor training, predictor HPO and parallel-launcher command-line help entrypoints execute successfully.
- The upload file set is checked for embedded credentials, machine-specific absolute paths and unintended binary/data files.

These checks validate packaging and software execution. They do not repeat full research training, reproduce reported accuracy, or perform CFD validation. Data and weights used for local compatibility checks are not uploaded.
