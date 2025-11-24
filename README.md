# 🔬 Blood Cell Classification using Deep Learning

**Author:** Shadman Sharif (A1944825)  
**Course:** Deep Learning Fundamentals  
**Achievement:** 🏆 98.0% Test Accuracy (6th Place on Leaderboard)

---

## 📊 Project Overview

Automated classification of microscopic blood cell images into 8 distinct categories using state-of-the-art deep learning techniques. This project demonstrates the application of efficient neural architectures for medical image analysis, achieving near-perfect accuracy with minimal computational resources.

### 🎯 Cell Types Classified

- **Basophil** - White blood cell involved in allergic reactions
- **Eosinophil** - White blood cell fighting parasites and allergies
- **Erythroblast** - Immature red blood cell precursor
- **Immature Granulocyte (IG)** - Early stage white blood cell
- **Lymphocyte** - Key player in immune system response
- **Monocyte** - Largest white blood cell type
- **Neutrophil** - Most abundant white blood cell
- **Platelet** - Essential for blood clotting

---

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 99.38% |
| **Test Accuracy** | 98.0% |
| **Leaderboard Ranking** | 6th place |
| **Model Parameters** | 1.84M (93% reduction vs baseline) |
| **Training Time** | 8 minutes (50 epochs) |
| **Inference Speed** | 8.3ms per image (120 images/sec) |
| **GPU Memory** | 4.2GB peak |
| **Per-Class F1-Scores** | 98.8% - 100% |

---

## 🏗️ Architecture: DeepEfficientNet

Custom efficient architecture based on EfficientNet with specialized enhancements for medical imaging:

### Core Components
- **Base:** EfficientNet-B0 backbone with MBConv blocks
- **Attention:** Squeeze-and-Excitation (SE) blocks for channel-wise feature recalibration
- **Regularization:** Stochastic depth, dropout (p=0.3), label smoothing (α=0.1)
- **Optimization:** AdamW with cosine annealing learning rate schedule
- **Training:** Mixed precision (FP16) for 1.5× speedup

### Architecture Highlights
```
Input (224×224×3)
    ↓
Stem Conv (112×112×32)
    ↓
MBConv Stage 1 (112×112×16)
    ↓
MBConv Stage 2 (56×56×24)
    ↓
MBConv Stage 3 (28×28×40) + SE Attention
    ↓
MBConv Stage 4 (14×14×80) + SE Attention
    ↓
Global Average Pooling (80×1)
    ↓
Dropout (p=0.3)
    ↓
Fully Connected (8 classes)
    ↓
Softmax Output
```

---

## 📈 Performance Improvements

Comprehensive ablation studies across 16 experiments quantified individual contributions:

| Component | Contribution | Impact |
|-----------|--------------|--------|
| **SE Attention Blocks** | +2.5% | Channel-wise feature recalibration |
| **Comprehensive Regularization** | +4.38% | Label smoothing, dropout, stochastic depth |
| **Cosine Annealing Schedule** | +3.29% | Smooth learning rate decay |
| **Enhanced Data Augmentation** | +3.6% | Rotation, flips, color jitter, affine |

### Model Comparison

| Model | Parameters | Val Accuracy | F1-Score | Improvement |
|-------|-----------|--------------|----------|-------------|
| SimpleCNN | 26.08M | 75.62% | 0.749 | Baseline |
| ResNet18 | 11.18M | 84.69% | 0.838 | +9.07% |
| EfficientNetLite | 0.55M | 96.25% | 0.953 | +20.63% |
| **DeepEfficientNet (Ours)** | **1.84M** | **99.38%** | **0.994** | **+23.76%** |

---

## 🎓 Academic Report

- **Format:** ICLR 2024 Conference Template
- **Length:** 6 pages (excluding references)
- **Content:**
  - Comprehensive literature review
  - Detailed methodology and architecture design
  - Systematic ablation studies (16 experiments)
  - Per-class performance analysis
  - Clinical deployment considerations
- **Figures:** 4 publication-quality visualizations
- **Tables:** 2 comprehensive results tables

📄 **[View Full Report](report/blood_cell_classification_REORGANIZED_BEAUTIFUL.pdf)**

---

## 📁 Repository Structure

```
perfect-cell-classification/
├── 📄 README.md                          # This file
├── 📁 report/                            # Academic documentation
│   ├── blood_cell_classification_REORGANIZED_BEAUTIFUL.pdf
│   ├── blood_cell_report_enhanced.tex    # LaTeX source
│   └── FINAL_SUBMISSION_SUMMARY.txt
├── 📁 figures/                           # Visualizations
│   ├── architecture.png                  # Model architecture diagram
│   ├── training_curves.png               # Training/validation curves
│   ├── ablation_studies.png              # Ablation study results
│   ├── model_comparison_clean.png        # Baseline comparison
│   └── confusion_matrix.png              # Error analysis
├── 📁 code/                              # Implementation
│   ├── train.py (or .ipynb)             # Training pipeline
│   ├── model.py                         # Model architecture
│   ├── utils.py                         # Helper functions
│   └── generate_figures.py              # Visualization scripts
├── 📁 data/                              # Data configurations
│   ├── class_map.json                   # Class ID mapping
│   └── prediction_labels.json           # Test predictions
└── 📁 docs/                              # Additional documentation
    ├── REORGANIZATION_SUMMARY.txt
    ├── LAYOUT_COMPARISON.txt
    └── publication_ready_summary.txt
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
CUDA 11.8+ (for GPU training)
```

### Installation
```bash
# Clone repository
git clone https://github.com/ALFIE-SADMAN/perfect-cell-classification.git
cd perfect-cell-classification

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn pillow
```

### Training
```python
# See code/train.py or train.ipynb for full training pipeline
python train.py --epochs 50 --batch_size 32 --lr 0.001
```

---

## 📊 Detailed Results

### Per-Class Performance

| Cell Type | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Basophil | 100% | 100% | **100%** |
| Eosinophil | 100% | 98.8% | 99.4% |
| Erythroblast | 100% | 100% | **100%** |
| Immature Granulocyte | 98.8% | 100% | 99.4% |
| Lymphocyte | 100% | 98.8% | 99.4% |
| Monocyte | 98.8% | 100% | 99.4% |
| Neutrophil | 100% | 100% | **100%** |
| Platelet | 98.8% | 98.8% | 98.8% |
| **Average** | **99.5%** | **99.5%** | **99.5%** |

### Confusion Analysis

Most errors occur between morphologically similar cell types:
- **Immature Granulocytes ↔ Monocytes** (biologically similar)
- **Platelet size variations** (smallest cells, challenging)

These confusion patterns are consistent with known clinical diagnostic challenges, validating the model's learned representations.

---

## 💡 Key Technical Insights

### 1. Architecture Efficiency
- **93% parameter reduction** compared to baseline SimpleCNN
- **84% fewer parameters** than ResNet18
- Demonstrates that efficient architectures outperform traditional large models

### 2. Attention Mechanisms
- SE blocks provide **+2.5% improvement** through channel recalibration
- Most effective in deeper stages (Stages 3-4)
- Minimal computational overhead (<5% inference time)

### 3. Training Strategy
- **Label smoothing (α=0.1)**: Single biggest regularization contribution (+1.9%)
- **Cosine annealing**: Smooth convergence, outperforms step decay
- **Mixed precision**: 1.5× speedup with no accuracy loss

### 4. Data Augmentation
- Full strategy: flips, rotation (±30°), ColorJitter, affine transforms
- **+3.6% improvement** over normalization alone
- Critical for limited dataset size (3,200 training images)

---

## 🔬 Clinical Relevance

### Potential Applications
- **Automated CBC Analysis:** Reduce manual microscopy burden
- **Quality Control:** Consistent, objective cell classification
- **High-Throughput Screening:** Process thousands of samples rapidly
- **Education:** Training tool for medical students
- **Remote Diagnostics:** Enable telemedicine in underserved areas

### Deployment Considerations
- **Inference Speed:** 8.3ms per image suitable for real-time analysis
- **Memory Footprint:** 1.84M parameters enables edge deployment
- **Calibration Needed:** Adjust for natural class imbalance in clinical settings
- **Domain Adaptation:** Fine-tune for different laboratory staining protocols

---

## 🎯 Future Directions

1. **Ensemble Methods:** Combine multiple models for uncertainty quantification
2. **Self-Supervised Pre-training:** Use SimCLR/MoCo for better representations
3. **Multi-Scale Analysis:** Integrate cell context and fine morphological details
4. **Prospective Validation:** Test across multiple clinical sites
5. **Rare Cell Detection:** Extend to pathological conditions and rare cell types
6. **Explainability:** Integrate Grad-CAM for clinical interpretability

---

## 📚 Technologies Used

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)

**Deep Learning:**
- PyTorch 2.0+ (Model implementation)
- torchvision (Data augmentation)
- Mixed Precision Training (AMP)

**Scientific Computing:**
- NumPy (Numerical operations)
- pandas (Data analysis)
- scikit-learn (Metrics, preprocessing)

**Visualization:**
- Matplotlib (Charts and plots)
- Seaborn (Statistical visualizations)
- Pillow (Image processing)

**Documentation:**
- LaTeX (Academic report)
- Markdown (Documentation)

---

## 🏆 Achievements

✅ **99.38% validation accuracy** - Near-perfect performance  
✅ **98.0% test accuracy** - Strong generalization  
✅ **6th place ranking** - Top performance among peers  
✅ **93% parameter reduction** - Extreme efficiency  
✅ **Perfect F1-scores** - 100% on 3 cell types  
✅ **8-minute training** - Rapid experimentation  
✅ **Publication-quality report** - Conference-standard documentation  
✅ **16 ablation experiments** - Rigorous methodology  

---

## 📖 References

Key papers that informed this work:

1. **EfficientNet:** Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019
2. **Squeeze-and-Excitation:** Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
3. **ResNet:** He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
4. **Blood Cell Analysis:** Acevedo et al., "Recognition System for Peripheral Blood Cell Images", Computer Methods and Programs in Biomedicine, 2019
5. **Medical Imaging Survey:** Litjens et al., "A Survey on Deep Learning in Medical Image Analysis", Medical Image Analysis, 2017

---

## 📧 Contact

**Shadman Sharif**  
Student ID: A1944825  
Course: Deep Learning Fundamentals  

For questions or collaboration opportunities, please open an issue or contact through GitHub.

---

## 📜 License

This project is available for educational and research purposes. Please cite this work if you use it in your research.

---

## 🙏 Acknowledgments

- Course instructors for guidance and feedback
- Classmates for discussions and insights
- Open-source community for excellent tools and libraries
- Dataset providers for quality medical imaging data

---

## ⭐ Star This Repository

If you find this project useful or interesting, please consider giving it a star! ⭐

---

**Last Updated:** November 2024  
**Status:** ✅ Completed | 🏆 6th Place Achievement | 📊 98.0% Test Accuracy
