# Self-diffusion for Solving Inverse Problems

**NeurIPS 2025** | [OpenReview](https://openreview.net/forum?id=5g9qls1V7Q) | [PDF](https://arxiv.org/pdf/2510.21417)

**Authors**: Guanxiong Luo, Shoujin Huang, Yanlong Yang

**TL;DR**: Self-diffusion solves inverse problems without the need of pretrained generative models via a self-contained iterative process that alternates between noising and denoising steps to progressively refine its estimate of the solution.

**Keywords**: inverse problems, computational imaging, image reconstruction, diffusion models

![overview](https://arxiv.org/html/2510.21417v1/figs/1.png)

This repository contains the implementation of Self-diffusion for solving inverse problems. The code is organized as follows:

- `sdi.py`: Main implementation of the Self-diffusion algorithm.
- `denoise.py`: Denoising network and training utilities.
- `utils.py`: Helper functions for data loading and processing.
- `simulation.py`: Simulation scripts for generating synthetic data.
- `run_*.sh`: Bash scripts to run experiments on different tasks (MRI, general inverse problems, hyperparameter tuning, etc.).
- `net/`: Neural network architectures (UNet, extra deep UNet, etc.).
- `misc/`: Miscellaneous files including sample data and images.

## Quick Start

To run Self-diffusion on a sample inverse problem (e.g., MRI reconstruction), use:

```bash
./run_mri.sh
```

Or for a general inverse problem:

```bash
./run_general.sh
```

## Citation

If you use this code or find the paper useful, please cite:

```bibtex
@inproceedings{
  luo2025selfdiffusion,
  title={Self-diffusion for Solving Inverse Problems},
  author={Guanxiong Luo and Shoujin Huang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=5g9qls1V7Q}
}
```
or

```
@article{luo2025selfdiffusionsolvinginverseproblems,
      title={Self-diffusion for Solving Inverse Problems}, 
      author={Guanxiong Luo and Shoujin Huang and Yanlong Yang},
      year={2025},
      eprint={2510.21417},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.21417}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, please contact the authors via OpenReview or open an issue in this repository.
