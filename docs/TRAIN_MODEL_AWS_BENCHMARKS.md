# Train a Model with Amazon Web Services (AWS) Benchmarks

This markdown benchmarks the various available AWS machines for training.

These benchmarks were completed on
[this commit](https://github.com/wellsaid-labs/Text-to-Speech/pull/237/commits/03339c83914bb2a4e8503526cfcbed9036b0b679).

## Spectrogram Model Training

| Comet                            | Machine Type  | GPU  | # GPU | GPU RAM | Spot Request \$/hr | Steps Per Second | \$/step |
| -------------------------------- | ------------- | ---- | ----- | ------- | ------------------ | ---------------- | ------- |
| 1326eb9879404cf5ab8796bb89a07896 | g4dn.12xlarge | T4   | 4     | 64      | 1.1736             | ~0.7             | 0.00046 |
| 274d99d28f964f029196a48d803b8931 | g3.16xlarge   | M60  | 4     | 32      | 1.3680             | ~0.475           | 0.00080 |
| 59c999963ebd4a50aa410106e4031d20 | p2.8xlarge    | K80  | 8     | 96      | 2.16               | ~0.45            | 0.00133 |
| b8260e607b4a4febafb4f3e15f3c79b2 | p3.8xlarge    | V100 | 4     | 64      | 3.6720             | ~0.7             | 0.00146 |

The GPUs that we evaluated all had at least 32 gigabytes of RAM, the minimum required to
train the spectrogram model without making any code changes.

## Signal Model Training

| Comet                            | Machine Type  | GPU  | # GPU | GPU RAM | Spot Request \$/hr | Steps Per Second | \$/step |
| -------------------------------- | ------------- | ---- | ----- | ------- | ------------------ | ---------------- | ------- |
| 395d52ef56c24f6283a4b4817d624353 | g4dn.12xlarge | T4   | 4     | 64      | 1.1736             | ~1.7             | 0.00019 |
