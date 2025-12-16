## TianXing-TC: Tropical Cyclone Forecasting System using Large Weather Models

This repository contains the official implementation of the TianXing-TC probabilistic forecasting system, designed to significantly enhance Tropical Cyclone (TC) intensity and track prediction accuracy.

### Project Overview

TianXing-TC is a novel system built upon the powerful machine learning-based global weather model, **TianXing**. It functions as a post-processing model that integrates large ensemble forecasts of environmental field trajectories with diverse data sources, including satellite cloud pattern images and statistical factors. The system addresses the common issue of intensity underestimation in raw Large Weather Model (LWM) forecasts.

The system comprises two main components:
1.  **Trajectories Generation Network**: Generates ensemble forecasts of environmental field trajectories centered on the TC's eye locations based on TianXing.
2.  **TC Predicting Network**: Corrects biases in TianXing's predictions by integrating the ensemble trajectories, satellite cloud pattern images, and statistical factors.

### Key Performance

Evaluated against current official forecasts for the North Atlantic (ATLN), Eastern North Pacific (EPAC), and Western North Pacific (WPAC) basins over the 2018-2020 period, TianXing-TC consistently demonstrates superior performance, especially at longer lead times.

* **Intensity Prediction**: TianXing-TC consistently outperforms official forecasts across all basins. At a 96-hour lead time, it achieves substantial Mean Absolute Error (MAE) reductions compared to official forecasts: **18.8%** in the WPAC basin (14.48 kt vs 17.84 kt), **14.9%** in the ATLN basin (11.44 kt vs 13.45 kt), and **12.1%** in the EPAC basin (15.36 kt vs 17.48 kt).
* **Track Prediction**: The system competes closely and demonstrates notable improvements at medium- to long-range predictions. It reduces the 96-hour MAE by **22.2%** in the WPAC basin (115.88 n mi vs 148.89 n mi) compared to the JTWC official forecast. In the ATLN basin, it achieves a lower MAE at 96 hours (113.81 n mi vs 133.09 n mi).

### Code and Data

The partial source code (PyTorch) for the TianXing-TC system is available at this repository.
The data utilized in this study is derived from publicly available sources:
* **TC Observations**: Best track data from IBTRACS (including NHC for ATLN/EPAC and JTWC for WPAC).
* **Environmental Data**: ERA5 Reanalysis data.
* **Satellite Imagery**: GridSat-B1 archive.

### Citation

If you use this work, please cite our paper:

```bibtex
@article{Yuan2025TianXingTC,
  title={Enhancing Tropical Cyclone Prediction through Large Weather Model Ensemble Forecasts and Multi-Source Data Integration},
  author={Yuan, Shijin and Wang, Guansong and Mu, Bin and Zhou, Feifan and Li, Hao and Zhang, Yake},
  journal={JAMES},
