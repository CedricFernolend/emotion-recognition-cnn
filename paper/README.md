# Paper: Progressive Attention Mechanisms for Facial Expression Recognition

## Compiling the Paper

### Option 1: Command Line
```bash
cd paper
chmod +x compile.sh
./compile.sh
```

### Option 2: Overleaf
1. Upload `emotion_recognition_paper.tex` to [Overleaf](https://www.overleaf.com)
2. Click "Recompile"
3. Download the PDF

### Option 3: Manual
```bash
pdflatex emotion_recognition_paper.tex
pdflatex emotion_recognition_paper.tex  # Run twice for references
```

## Adding Figures

To include the generated visualizations, add these to the LaTeX file:

```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{../results/comparison/training_curves_comparison.png}
    \caption{Training and validation accuracy curves for V1, V2, and V3 models.}
    \label{fig:training_curves}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{../results/comparison/confusion_matrices_grid.png}
    \caption{Confusion matrices for all three model versions.}
    \label{fig:confusion}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\columnwidth]{../results/architecture_diagrams/architecture_comparison.png}
    \caption{Architecture comparison of V1, V2, and V3 models.}
    \label{fig:architecture}
\end{figure}
```

## Paper Structure

| Section | Pages | Content |
|---------|-------|---------|
| Abstract | 0.3 | Summary of work and key results |
| Introduction | 0.8 | Problem motivation, challenges, contributions |
| Related Work | 1.0 | FER methods, attention mechanisms, class imbalance |
| Dataset | 0.5 | FER2013 overview, preprocessing, augmentation |
| Methodology | 1.5 | V1, V2, V3 architectures in detail |
| Experiments | 2.0 | Results, per-class analysis, ablation studies |
| Discussion | 0.7 | Effectiveness, limitations, real-time application |
| Conclusion | 0.3 | Summary and future work |
| References | 1.0 | 26 scientific citations |

**Total: ~8 pages**

## Key Citations Included

- FER2013 dataset (Goodfellow et al., 2013)
- ResNet (He et al., 2016)
- SENet (Hu et al., 2018)
- CBAM (Woo et al., 2018)
- Grad-CAM (Selvaraju et al., 2017)
- Vision Transformers (Dosovitskiy et al., 2020)
- Focal Loss (Lin et al., 2017)
- Label Smoothing (Müller et al., 2019)
