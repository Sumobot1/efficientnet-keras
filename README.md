# efficientnet-keras
Efficientnet implementation using Keras

Code is heavily based on https://github.com/qubvel/efficientnet

Changes from the original:
- Refactoring
- I used a different calculation for total_network_blocks - I think the ref implementation is wrong? (Relevant PR: https://github.com/qubvel/efficientnet/pull/130)
- Added lots of comments :)
- Checked performance of my implementation vs PyPi package on CIFAR10
- Added visualization for conv initializer
