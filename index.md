# CarboNeXT and CarboFormer: Dual Semantic Segmentation Architectures for CO₂ Emission Detection

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2506.05360-b31b1b.svg)](https://arxiv.org/abs/2506.05360)
[![Code](https://img.shields.io/badge/Code-GitHub-black.svg)](https://github.com/taminulislam/carbonext_and_carboformer)
<!-- [![Paper](https://img.shields.io/badge/Paper-NeurIPS%202025-blue.svg)](neurips_2025.pdf) -->
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**🚀 Real-time CO₂ emission detection and quantification using Optical Gas Imaging (OGI)**

</div>

## 📝 Overview

We introduce **CarboNeXT** and **CarboFormer**, two novel semantic segmentation frameworks designed for detecting and quantifying CO₂ emissions using Optical Gas Imaging (OGI) technology. Our dual-architecture approach addresses both high-performance monitoring and resource-constrained deployment scenarios, making CO₂ detection accessible across diverse applications from environmental sensing to precision livestock management.

### 🎯 Key Highlights

- **CarboNeXT**: High-performance model achieving **88.46% mIoU** on CCR dataset and **92.95% mIoU** on RTA dataset at **60.95 FPS**
- **CarboFormer**: Lightweight variant with only **5.07M parameters** achieving **84.68 FPS** with competitive performance (**84.88% mIoU** on CCR, **92.98%** on RTA)
- **Real-time capabilities** for environmental monitoring and precision agriculture
- **Effectiveness in challenging low-flow scenarios** for accurate CO₂ emission analysis
- **Dual dataset contribution**: CCR (Controlled Carbon dioxide Release) and RTA (Real Time Ankom) datasets

## 🏗️ Architecture Overview

### CarboNeXT Architecture
![CarboNeXT Framework](./figures/carbonext.pdf)

**CarboNeXT** employs a sophisticated encoder-decoder structure featuring:
- **Multi-Scale Context Aggregation Network (MSCAN)** encoder with 4 progressive stages
- **Unified Perceptual Parsing Head (UPerHead)** with Pyramid Pooling Module (PPM) and Feature Pyramid Network (FPN)
- **Auxiliary FCN head** for enhanced supervision during training
- **Multi-scale attention mechanisms** with stage-specific kernel transformations

### CarboFormer Architecture  
![CarboFormer Framework](./figures/carbonformer.pdf)

**CarboFormer** offers an efficient lightweight design with:
- **Mix Vision Transformer (MViT)** backbone for hierarchical feature extraction
- **Light Harmonic Aggregation Module (LightHAM)** for efficient multi-scale feature processing
- **Overlap Patch Merging** to preserve spatial continuity
- **Optimized attention mechanisms** for resource-constrained platforms

## 📊 Datasets

![Dataset Overview](./figures/dataset_ccr.pdf)

### 🧪 Controlled Carbon dioxide Release (CCR) Dataset
- **19,731 images** (640×480 pixels) captured using FLIR G343 OGI camera
- **Systematic CO₂ releases** at flow rates from 10-100 SCCM in controlled laboratory conditions
- **Multiple visualization modes**: White hot, Black hot, and Lava
- **Diverse experimental conditions** with controlled temperature, pressure, and distance parameters

![CCR Dataset Samples](./figures/dataset.pdf)

### 🐄 Real Time Ankom (RTA) Dataset  
- **613 images** (640×480 pixels) from dairy cow rumen fluid CO₂ emissions
- **ANKOM gas production system** for realistic agricultural scenarios
- **pH variations** (6.5 to 5.0) generating diverse CO₂/CH₄ concentrations
- **Binary segmentation labels** (gas vs. background) for practical applications

![RTA Dataset and Performance](./figures/rta_result_graph.pdf)

## 🚀 Performance Results

### Quantitative Performance

| Model | Category | mIoU (CCR) | mIoU (RTA) | FPS | Parameters | FLOPs |
|-------|----------|------------|------------|-----|------------|-------|
| **CarboNeXT** | Heavy-weight | **88.46%** | **92.95%** | 60.95 | 31.71M | 214G |
| **CarboFormer** | Light-weight | **84.88%** | **92.98%** | 84.68 | 5.07M | 11.39G |
| SegFormer-B5 | Heavy-weight | 86.63% | 92.84% | 22.16 | 81.98M | 74.63G |
| SegFormer-B0 | Light-weight | 83.36% | 92.35% | 121.06 | 3.72M | 7.92G |

### Key Achievements
- **🏆 State-of-the-art performance** in both heavy-weight and light-weight categories
- **⚡ Real-time inference** capabilities suitable for continuous monitoring
- **🎯 Superior accuracy** particularly in challenging low-flow emission scenarios
- **💡 Efficient resource utilization** making deployment feasible on programmable drones

### Qualitative Results
![CCR Dataset Results](./figures/ccr_result2.pdf)

The qualitative results demonstrate superior segmentation performance across different CO₂ flow rates (10, 70, and 100 SCCM), showing improved boundary preservation and detection accuracy compared to existing methods.

![Additional Results Comparison](./figures/ccr_result3.pdf)

## 🛠️ Technical Implementation

### Equipment Specifications
- **FLIR G343 OGI Camera**: Specialized cooled mid-wave infrared (MWIR) system
  - **Spectral Range**: 4.2-4.4 μm (aligned with CO₂ absorption band at 4.3 μm)
  - **Resolution**: 320×240 pixel quantum detector
  - **Sensitivity**: <15 mK NETD, 30-50 ppm-m NECL for CO₂
  - **Detection Range**: Up to several hundred meters

### Mask Generation Pipeline
- **Semi-automated annotation** using differential background modeling
- **Temporal averaging** for robust background subtraction
- **Adaptive thresholding** calibrated per flow rate category
- **Watershed algorithm** for precise boundary delineation
- **Morphological post-processing** with validation checks

## 📈 Applications

### 🌍 Environmental Monitoring
- **Industrial leak detection** with real-time alerting capabilities
- **Atmospheric CO₂ tracking** for climate research
- **Regulatory compliance** monitoring for emission standards

### 🚜 Precision Agriculture
- **Livestock emission quantification** for carbon footprint assessment
- **Rumen health monitoring** through CO₂ pattern analysis
- **Farm-level carbon accounting** for sustainable agriculture practices

### 🚁 Drone-Based Monitoring
- **Mobile emission surveys** using CarboFormer's lightweight architecture
- **Remote area monitoring** in challenging access locations
- **Large-scale agricultural surveying** with autonomous systems

## 🔬 Research Contributions

1. **Novel Architectures**: Introduction of CarboNeXT and CarboFormer with specialized designs for CO₂ detection
2. **Comprehensive Datasets**: Creation of CCR and RTA datasets for systematic evaluation
3. **Performance Benchmarking**: Rigorous comparison against state-of-the-art segmentation methods
4. **Real-world Applicability**: Demonstration of effectiveness across diverse emission scenarios
5. **Efficiency Optimization**: Balanced trade-offs between accuracy and computational requirements

## 📋 Installation & Usage

### Prerequisites
```bash
# Create conda environment
conda create -n carbonext python=3.8
conda activate carbonext

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install MMSegmentation dependencies
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0,<2.2.0"
```

### Model Training
```bash
# Train CarboNeXT
python tools/train.py local_configs/hw_model/carbonext/carbonext_config.py

# Train CarboFormer  
python tools/train.py local_configs/hw_model/carboformer/carboformer_config.py
```

### Model Evaluation
```bash
# Evaluate on test set
python tools/test.py config_file checkpoint_file --eval mIoU

# Benchmark performance
python tools/analysis_tools/benchmark.py config_file checkpoint_file

# Calculate FLOPs
python tools/analysis_tools/get_flops.py config_file --shape 512 512
```

## 📊 Experimental Analysis

### Comprehensive Results Visualization
![Detailed Results Analysis](./figures/rta_result.pdf)

### Ablation Studies
Our comprehensive ablation experiments validate key architectural choices:

**CarboNeXT Components:**
- Multi-scale feature fusion provides consistent improvements
- 4-stage decoder architecture optimizes accuracy-efficiency balance
- 512-channel dimension maintains optimal performance
- Auxiliary head supervision is crucial for stable training

**CarboFormer Optimizations:**
- Strategic transformer depth reduction improves efficiency
- 4-stage decoder enhances lightweight performance
- Auxiliary supervision contributes to better feature learning

### Qualitative Results
Visual comparisons demonstrate superior performance across:
- **Boundary precision** in complex gas dispersion patterns
- **Fine-grained detail preservation** in low-flow scenarios  
- **Consistent segmentation quality** across varying experimental conditions
- **Robust performance** in challenging real-world environments

![Performance Analysis](./figures/ccr_result.pdf)

## 🚀 Future Directions

### Near-term Improvements
- **Dataset expansion** through direct farm measurements with varying pH conditions
- **Temporal modeling** for dynamic gas pattern analysis
- **Knowledge distillation** to further optimize CarboFormer efficiency

### Long-term Vision
- **Multi-gas detection** extending beyond CO₂ to CH₄ and other greenhouse gases
- **Integration with IoT systems** for comprehensive environmental monitoring
- **Edge deployment optimization** for autonomous monitoring systems

## 📖 Citation

If you use CarboNeXT or CarboFormer in your research, please cite our paper:

```bibtex
@article{islam2025carbonext,
  title={CarboNeXT and CarboFormer: Dual Semantic Segmentation Architectures for Detecting and Quantifying Carbon Dioxide Emissions Using Optical Gas Imaging},
  author={Islam, Taminul and Sarker, Toqi Tahamid and Embaby, Mohamed G and Ahmed, Khaled R and AbuGhazaleh, Amer},
  journal={arXiv preprint arXiv:2506.05360},
  year={2025}
}
```

## 👥 Authors

**Taminul Islam¹**, **Toqi Tahamid Sarker¹**, **Mohamed G Embaby²**, **Khaled R Ahmed¹**, **Amer AbuGhazaleh²**

¹School of Computing, ²School of Agricultural Sciences  
Southern Illinois University, Carbondale, USA

## 📧 Contact

For questions, collaborations, or technical support:
- 📧 Email: {taminul.islam, toqitahamid.sarker, mohamed.embaby, khaled.ahmed, aabugha}@siu.edu
- 📄 Paper: [arXiv:2506.05360](https://arxiv.org/abs/2506.05360)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the Southern Illinois University for providing computational resources and the dairy farm collaborators for enabling real-world data collection. Special appreciation to the computer vision and agricultural engineering communities for their valuable feedback and support.

---

<div align="center">
<b>🌱 Contributing to sustainable agriculture and environmental monitoring through advanced AI 🌱</b>
</div> 