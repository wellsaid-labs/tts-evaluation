# Train a Model with Google Cloud Platform (GCP) Benchmarks

This markdown benchmarks the various available GCP machines for training.

These benchmarks were completed on
[this commit](https://github.com/wellsaid-labs/Text-to-Speech/pull/302/commits/b50715e29a81e4d13a55e80213534e457e8656e4).

## Spectrogram Model Training

| Comet                            | Machine Type     | GPU | # GPU | GPU RAM | Spot Request \$/hr | Steps Per Second | \$/step |
| -------------------------------- | ---------------- | --- | ----- | ------- | ------------------ | ---------------- | ------- |
| dc7df549242d4a3ebf043761d8678631 | custom-24-73728  | T4  | 4     | 64      | 0.665              | ~1.19            | 0.00015 |
| 797641234fb34664941b770bf7aa965c | custom-32-98304  | T4  | 4     | 64      | 0.74               | ~1.25            | 0.00016 |
| fa07c8c28ad34f359c2f62c4000d0093 | custom-48-199680 | T4  | 4     | 64      | 0.935              | ~1.36            | 0.00019 |

