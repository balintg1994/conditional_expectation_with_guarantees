# Conditional Expectation with Numerical Guarantees

Are you tired of black-box machine learning algorithms that lack transparency and fail to provide trustworthy predictions? Look no further!

This repository presents a novel approach developed by [Prof. Patrick Cheridito](https://people.math.ethz.ch/~patrickc/) and myself for computing conditional expectations with numerical guarantees.

Using an alternative expected value representation of the minimal L2 distance between Y and f(X) over all Borel measurable functions f, we provide guarantees for the accuracy of any numerical approximation of a given conditional expectation. We illustrate the method by assessing the quality of numerical approximations to different high-dimensional nonlinear regression problems.


For more details on the theoretical background and methodology used in this library, please refer to the following research paper:

<em>Computation of conditional expectations with guarantees</em><br>
<a href="https://arxiv.org/abs/2112.01804">arXiv Preprint 2112.01804</a><br>
<a href="https://link.springer.com/article/10.1007/s10915-023-02130-8">Journal of Scientific Computing</a>
95(12), 2023 Springer

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Running Code

### Installation

Start by installing the necessary dependencies. You can do this by running the following command:

```bash
pip install -r requirements.txt
```


#### Usage

To execute the experiment, use the following command-line input:

```bash
python3 linear_example.py
```

#### Acknowledgement

Speacial thanks to OpenAI's ChatGPT for helping with writing, refactoring and improving the souce code. 


#### Citation

If you use this work in your research or publication, please cite the following paper:

```commandline
@article{cheridito2023computation,
  title={Computation of conditional expectations with guarantees},
  author={Cheridito, Patrick and Gersey, Balint},
  journal={Journal of Scientific Computing},
  volume={95},
  number={1},
  pages={12},
  year={2023},
  publisher={Springer}
}
```
