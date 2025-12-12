# **Unsupervised Counterfeit Detection of Pokemon Trading Cards via Patch-Based Anomaly Detection**

# Introduction

## 1 Problem Statement

**This project attempts to study counterfeit detection of Pokemon trading cards, and posits that anomaly detection is a natural way to approach this problem.**

The project begins with a Kaggle dataset containing "real" and "fake" Pokemon cards, and the dataset author frames the dataset as suited for standard binary supervised classification. However, the premise of this project disagrees, and instead aligns with the notion that the class of problem is best approached as an anomaly detection task, where "real" cards compose the _normal_ class, and "fake" cards compose the _anomalous_ class.

Anomaly detection is a family of methods that learns the distribution of _normal_ data only and then flags samples which deviate significantly from the learned manifold of _normal_ data as anomalous. Counterfeit detection for Pokemon cards is naturally suited to an anomaly detection setup because genuine Pokemon cards are stable, well-defined, and easy to sample, while counterfeit Pokemon cards are open (not well-defined), relatively much more sparsely sampled, and constantly evolving. Supervised models, such as supervised CNNs, which train on both fake and real labels can only learn the subset of "fake" cards present in their training data, and new counterfeit designs which differ from those seen during training are likely both to appear and be misclassified by the model. Thus, such a model requires frequent retraining on freshly labeled samples, which may or may not always exist for new kinds of counterfeit cards. However, anomaly detection models only require training on the _normal_ class; thus, given that genuine Pokemon cards vary very infrequently over time, the model realistically only needs to train once on a relatively easy-to-acquire dataset in order to learn how to detect any kind of counterfeit card.

The question to be addressed then is whether an unsupervised learning method which only trains on the _normal_ class is expected to achieve sufficient performance, and this will be tested empirically.

## 2 Scope and Pipeline

To answer this question, the scope of this project is to implement and evaluate how well an anomaly detection pipeline can separate real and fake Pokemon cards. Importantly, the creator of the dataset explicitly states that the cards are nontrivial for humans only moderately familiar with Pokemon cards to classify, which means that sufficient performance on this dataset indicates potential for aiding humans in counterfeit detection itself rather than simply allowing for automation.

![402](https://private-user-images.githubusercontent.com/13025381/526070916-8edaa748-c3b3-4cc6-ae33-e4ec9939d98c.JPG?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU1NzM2MTAsIm5iZiI6MTc2NTU3MzMxMCwicGF0aCI6Ii8xMzAyNTM4MS81MjYwNzA5MTYtOGVkYWE3NDgtYzNiMy00Y2M2LWFlMzMtZTRlYzk5MzlkOThjLkpQRz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjEyVDIxMDE1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWYxZTBmODg3NjBhODlmMDMzNjMwYzIwZDA3MTk1ZjdiYWUzNWU1MjI3NGI4MzBkYzI0NGZiMDEyYmFiZTQ4MWMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.bAg0ifNSBtkV4uu9GiAnVEIAsyvQdGELC0NYOBHWuE0)

_Figure 1. Example of a "real" Pokemon card from the dataset (402.JPG)_

![383](https://private-user-images.githubusercontent.com/13025381/526070732-e4879e91-1838-45f3-880c-7526f6cd255d.JPG?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU1NzM2MTAsIm5iZiI6MTc2NTU3MzMxMCwicGF0aCI6Ii8xMzAyNTM4MS81MjYwNzA3MzItZTQ4NzllOTEtMTgzOC00NWYzLTg4MGMtNzUyNmY2Y2QyNTVkLkpQRz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjEyVDIxMDE1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTI0ZWJkZGNmMDc0ZTJjZWM3YWI0ZWQ2OWRjYWNlYmViNjFhNDQ5ODdjY2ZlZjNjYTZiZWEyMzBkMjA5YWJlYjYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.9VqCUjPRTF91G3HKoXpvDEAeO52JpZm8bJbUIWziDSU)

_Figure 2. Example of a "fake" Pokemon card from the dataset (383.JPG)_

The pipeline of the project is as follows:

1. **Dataset Preparation**
    
    * The Kaggle dataset consists of images of the backs of Pokemon cards photographed in a consistent manner with good lighting and minimal background. The dataset labels are encoded in a CSV (0 for "fake", 1 for "real").
        
    * The pipeline assigns each "fake" card to the _anomalous_ class and each "real" card to the _normal_ class (0 for _normal_, 1 for _anomalous_ so the numerical classification is reversed relative to the dataset). An AD (anomaly detection) style directory layout is then constructed from the dataset (with train/normal, val/normal, val/anomalous, test/normal, test/anomalous), where the training data consists solely of real cards and the validation and test splits contain both real and fake cards in known proportions.
        
2. **Image Preprocessing**
    
    * Each of the images are loaded from the constructed AD dataset, read in the correct format (RGB) to prepare for producing tensors, and are normalized using ImageNet mean and standard deviation (best suited for PatchCore, the anomaly detection model which will be used)
        
    * Tensors are produced from these images of shape (3, H, W) which means that each image is represented using 3 channels of $H \times W$ matrices (R, G, B), which are appropriate for usage with PatchCore, which uses a pretrained CNN backbone.
        
3. **Train PatchCore (Anomaly Detection Model)**
    
    * Within PatchCore, the pretrained CNN backbone extracts multi-layer patch embeddings from the training images
        
    * A memory bank of patch features using coreset sampling are built to create a compact and representative set of embeddings
        
    * The backbone and memory bank are frozen after this stage
        
4. **Compute Anomaly Scores and Maps**
    
    * For each image in the validation and test splits, the model computes patch-wise distances to the learned normal class memory bank (this is how the model computes how distant an image's features are from the learned normal manifold)
        
    * A scalar image-level anomaly score is generated, along with a 2D anomaly heatmap which highlights how much each region of the image differs from what it encodes as a normal image
        
5. **Evaluate performance**
    
    * ROC-AUC is computed from the anomaly score and label pairs for each image in the validation split. This measures how well the anomaly scores rank images so that a randomly chosen anomalous image gets a higher score than a randomly chosen normal image
        
    * _J_ values are calculated for many possible cutoffs by computing _J = TPR - FPR_ and Youden's J is the maximum of these _J_ values. The cutoff with the largest _J_ value is chosen as the anomaly score threshold which best separates the data.
        
    * Using this threshold, the F1 score, which is the harmonic mean of precision (of the images the model called anomalous, how many were truly anomalous) and recall (of all truly anomalous images, how many did the model catch) computed by comparing the values of the scores with the best threshold produced using the Youden's J statistic.
        
    * These are calculated for the test split as well. The threshold calculated using the validation split is reused with the test split during one of the tests to avoid leaking test-set information into the threshold choice and to mimic real deployment; the F1 score here is a robust and honest estimate of how well this operating point transfers to unseen data.
        
    * The threshold and F1 score calculated when test-set information is leaked into threshold choice and the F1 score here is used to see the theoretical best separation the model can achieve with respect to separation.
        

# Experiment

## 1 Problem Formulation and Dataset Construction

* Let $\mathcal{X} = {0,\dots,255}^{H\times W \times 3}$ denote the space of RGB images.
    
* Let $x \in \mathcal{X}$ denote a card image.
    
* Let $y_{\text{raw}} \in {0,1}$ denote the original Kaggle label, with $y_{\text{raw}}=1$ for real and $y_{\text{raw}}=0$ for fake.
    
* Let $y \in {0,1}$ denote the anomaly-detection label, with $y=0$ for normal ("real") and $y=1$ for anomalous ("fake"), so $y = 1 - y_{\text{raw}}$.
    
* Let $\mathcal{D}_{train}={x_i \mid y_i=0}$ denote the training set, which defines what _normal_ looks like to the model
    
* Let $\mathcal{D}_{val}={x_j,y_j}$ denote the validation set, which is primarily used to choose a decision threshold
    
* Let $\mathcal{D}_{test}={x_k,y_k}$ which is used to evaluate the final performance
    

## 2 Preprocessing and Data Model

* **Resizing:** Each image is resized to $H=256,;W=256$
    
* **Pixel Normalization:** Each pixel is converted to a floating-point number between 0 and 1 according to $x'_{h,w,c} = \frac{x_{h,w,c}}{255}.$
    
* **ImageNet Normalization:** Each channel is normalized using standard ImageNet statistics (these values are standard values from literature)
  
$$
\tilde{x}_{h,w,c} = \frac{x'_{h,w,c} - \mu_c}{\sigma_c},\quad
\mu=(0.485, 0.456, 0.406), \quad 
\sigma=(0.229, 0.224, 0.225), \quad 
$$
        
* **Tensor Format Input for PyTorch:** The final input to the network after preprocessing is $\tilde{x} \in \mathbb{R}^{3 \times H \times W}.$
    
* **Mask:** No pixel-level masks exist. Thus, image masks are created by treating every pixel in anomalous images as anomalous and every pixel in normal images as non-anomalous. This is actually quite robust for the setup, because the images in the dataset have very little background and counterfeit Pokemon cards have fundamentally different printing processes which will likely create anomalous characteristics across the entire printing surface.
    
    * Let $x = \tilde{x}$ denote the normalized image tensor
        
    * Let $y \in {0,1}$ denote the anomaly label
        
    * Let $M$ denote the mask.
        
    * Each loaded item into the network is a triple $(x, y, M)$
        

$$M = \begin{cases}  
0_{1\times H \times W}, & y = 0 \\  
1_{1\times H \times W}, & y = 1  
\end{cases}$$

* The lack of pixel-level masks means that anomaly heatmaps later computed will indicate qualitative anomalousness rather than pixel-level anomalous detail detected
    

## 3 PatchCore

* PatchCore uses a pretrained CNN network (Wide-ResNet-50-2). The two standard options are ResNet18 and Wide-ResNet-50-2
    
* This experiment used Wide ResNet because this task benefits from high channel capacity, detecting fine texture, print artifacts, and small color deviations, so ResNet18 was found to be suboptimal for the task
    
* Additionally, Layers 2-4 were utilized because Layer 2 encodes fine local statistics like subtle printing differences, Layer 3 encodes "mesoscopic structure" like coherent but incorrect printing structure, and Layer 4 encodes global coherence, allowing locally normal but globally inconsistent prints.
    
* Layer 1 was discarded because it is too fine grained and added too much overhead and risk of overfitting/false positives to justify.
    
* Let $f$ be the CNN backbone and $\mathcal{L} = {\ell_2, \ell_3, \ell_4}$ be the layers from which features are extracted
    
* **Feature Layers:** Each layer produces a feature map
    

$$F^{(\ell)} = f^{(\ell)}(\tilde{x}) \in \mathbb{R}^{C_\ell \times H_\ell \times W_\ell}$$

* And flattening each spatial location produces patch vectors:
    

$$P^{(\ell)}(\tilde{x}) \in \mathbb{R}^{N_\ell \times C_\ell},\; N_\ell = H_\ell W_\ell$$

* **Patch Embedding Set:** Features from layers $\ell_2,\ell_3,\ell_4$ are then spatially aligned to a common resolution and concatenated at each corresponding location to form a single multi-scale embedding vector for every image patch.
    

$$Z(\tilde{x}) = \{ z_p \in \mathbb{R}^d \mid p \in \mathcal{P} \}$$

where each $z_p$ describes one patch of image $\tilde{x}$.

* **Memory Bank (Coreset)**: All patch embeddings are gathered from normal training images:
    

$$\mathcal{Z} = \bigcup_{x_i \in \mathcal{D}_{\text{train}}} Z(\tilde{x}_i)  
= \{ z_1, z_2, \dots, z_{N_p} \}$$

* The distance between two patch embeddings is calculated using the euclidean norm:
    

$$d(u,v) = \|u - v\|_2$$

* PatchCore selects a subset $M\subset \mathcal{Z}$ of size $|M| = \lfloor \rho N_p \rfloor$. This experiment chose $\rho=0.1$ as a good tradeoff between performance and computational overhead for this specific dataset and the hardware constraints. This is the _memory bank_ of normal patches. (According to the paper, patch embeddings are subsampled using greedy farthest-point coreset selection in feature space to preserve coverage of the normal manifold)
    
* **Patch-Level Anomaly Score:** To find the patch level anomaly score given a new test image $\tilde{x}$
    
    * Patch embeddings are computed using $Z(\tilde{x}) = {z_p}$
        
    * The nearest memory patch for embedding is calculated $m_{p,1} = \arg\min_{m \in M} d(z_p, m).$
        
    * The distance function is used to calculate how anomalous the image is: $a_p = d(z_p, m_{p,1}).$ In other words, how anomalous an image is is represented by how far the model "thinks" the image is from the learned manifold of _normal_ images in feature space. (According to the paper, Patch embeddings are $\ell_2$-normalized prior to nearest-neighbor distance computation.)
        
* **Heatmap:** Heatmaps are constructed by expanding the image back to full dimensions and producing a pixel-level image where brightness indicates how anomalous PatchCore "thinks" that patch is.
    
* **Image-Level Score:** The image-level score is computed by utilizing the score of the most anomalous patch. This makes sense because anomalies may be highly localized, and an anomalous image is seen as anomalous due to its anomalous areas, not just how "abnormal the image looks _on average_":
    

$$s(x) = \max_{p\in\mathcal{P}} a_p$$

[1] K. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, and P. Gehler, “Towards Total Recall in Industrial Anomaly Detection,” in _Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)_, 2022, pp. 14298–14308, doi: 10.1109/CVPR52688.2022.01392.

[2] “PatchCore — Anomalib documentation,” _Read the Docs_. [Online]. Available: [https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html](https://anomalib.readthedocs.io/en/v2.0.0/markdown/guides/reference/models/image/patchcore.html) (accessed Dec. 12, 2025)

## 4 Evaluation Metrics

* Evaluation metrics are calculated using anomaly scores and labels: ${(s_i,y_i)}_{i=1}^n,$
    
* Let $\tau$ denote the threshold.
    
* **Classification at Threshold:** The prediction rule is $\hat{y}_i(\tau) = \mathbf{1}[ s_i \ge \tau ]$ (indicator function). True positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) are all computed using this rule.
    
* **True Positive Ratio and False Positive Ratio:** These are computed accordingly:
    

$$\mathrm{TPR}(\tau) = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}$$ $$\mathrm{FPR}(\tau) = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}}$$

* **ROC and AUC**: The "receiver operating characteristic" or "ROC" curve plots $(\mathrm{TPR}(\tau), \mathrm{FPR}(\tau))$ as $\tau$ varies and the AUC is the area under this curve. The curve characterizes the trade-off between true-positive rate and false-positive rate across thresholds. The AUC is a scalar summary of this curve and equals the probability that a randomly chosen positive sample has a higher score than a randomly chosen negative sample. The following indicates how AUC generally informs perception of a model:
    
    * **0.5**: No discrimination (random chance).
        
    * **0.5 - 0.6**: Poor.
        
    * **0.6 - 0.7**: Fair/Poor (varies).
        
    * **0.7 - 0.8**: Moderate/Fair/Good (varies).
        
    * **0.8 - 0.9**: Good.
        
    * **0.9 - 1.0**: Excellent.
        
* **Youden’s J Statistic:** $J(\tau)$ computes at threshold $\tau$ the vertical distance between the ROC curve and the chance line (think the ROC curve of a "coin flip" style model), measuring how well positives are separated from negatives at that model according to $J(\tau) = \mathrm{TPR}(\tau) - \mathrm{FPR}(\tau).$
    
* **Best Threshold:** $\tau^*$ selects the decision threshold that maximizes this separation according to $\tau^\star = \arg\max_\tau J(\tau)$ such that true positives are maximized and false positives are minimized
    
* **Precision and Recall:** Precision measures the fraction of predicted positives that are correct according to $\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}$ and recall measures the fraction of true positives that are successfully detected according to $\mathrm{Recall} = \mathrm{TPR}$.
    
* **F1 Score:** The F1 score is found by taking the harmonic mean of precision and recall, which penalizes imbalance between the two according to
    

$$\mathrm{F1} = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$$

# Results

## 1 Quantities

* **Training/validation/test composition:** (all sets disjoint)
    
    * Train: 200 normal ("real") cards
        
    * Validation: 50 normal, 123 anomalous ("fake") cards
        
    * Test: 50 normal, 28 anomalous
        
* **Validation Performance:** (mixed normal and anomalous):
    

![image-20251212104551707](https://private-user-images.githubusercontent.com/13025381/526070373-648fee44-1552-4f00-8b85-facf4a22ad36.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU1NzM2MTAsIm5iZiI6MTc2NTU3MzMxMCwicGF0aCI6Ii8xMzAyNTM4MS81MjYwNzAzNzMtNjQ4ZmVlNDQtMTU1Mi00ZjAwLThiODUtZmFjZjRhMjJhZDM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjEyVDIxMDE1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJkYzE0MmMxNTRlMzg5YTMzZjA5OTE2ZTM3OWRkYWQwMDk2ZmNmYjhjOWY3NTY0ZmExNjUyYjY0NmZhYjBmYjUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.KZqiz_LPdI2BYKaGgI39XfQClKxYoKMrMZ9JPw0_7mg)

_Figure 3. Metrics for Validation Set_

* **Test Performance using best threshold from Validation**: (i.e. deployment setting)
    

![image-20251212104814092](https://private-user-images.githubusercontent.com/13025381/526070401-8a1f61e0-3ff2-417b-a2d8-f956374fba2f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU1NzM2MTAsIm5iZiI6MTc2NTU3MzMxMCwicGF0aCI6Ii8xMzAyNTM4MS81MjYwNzA0MDEtOGExZjYxZTAtM2ZmMi00MTdiLWEyZDgtZjk1NjM3NGZiYTJmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjEyVDIxMDE1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA0ZTJhNDYwMWEzYjYwMWM0YjgyMTlkMzhkMmNmZjU5YWQ3OGI5MDNlMWVjOWI1Y2IwN2U5YmNjZDgxMmI0N2ImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.v9x0eoQQ6HK8tX3s7Qb8vi-7BquoA6QYHoUE4Y6EZ3Y)

_Figure 4. Metrics for Test set using Validation threshold_

* **Test Performance using Test-set-optimized best threshold**: (theoretical best, i.e. "oracle upper bound")
    

![image-20251212104951800](https://private-user-images.githubusercontent.com/13025381/526070447-8732be1c-7096-427f-a5e4-098c75434889.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjU1NzM2MTAsIm5iZiI6MTc2NTU3MzMxMCwicGF0aCI6Ii8xMzAyNTM4MS81MjYwNzA0NDctODczMmJlMWMtNzA5Ni00MjdmLWE1ZTQtMDk4Yzc1NDM0ODg5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjEyVDIxMDE1MFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTg4YTg0ODk1ZWE1YzNiOGQxNzVkNzUyOTQ3OGMxMDc4YjNlZDgwMWU0ZGY1MmM3Nzc2MmY4ZWFmZDFkNTM1OGYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.bbNFlE-6hdcn0vzMIzbioMpGnZ1dQUdkFGnohmRiDV0)

_Figure 5. Metrics for Test set using Test-set optimized threshold_

## 2 Statistical Meaning

* Test set size:
    
    * $n_1 = 28$ counterfeits (positives),
        
    * $n_0 = 50$ genuine cards (negatives).
        
    * Observed AUC: $\hat{A} = 0.95$.
        
* Using the standard Hanley–McNeil asymptotic variance for AUC, this sample size yields a rough 95% confidence interval of [0.89,1.00]
    

$$\mathrm{Var}(\hat{A}) = \frac{ \hat{A}(1-\hat{A}) + (n_1 - 1)(Q_1 - \hat{A}^2) + (n_0 - 1)(Q_2 - \hat{A}^2) }{ n_1 n_0 }$$

* This is statistically high, and strongly suggests the results are not a fluke arising from noise:
    
    * The lower bound is far above random chance (0.5)
        
    * Even accounting for sampling variability, the model's ranking performance is in the "strong separation" category
        
    * There is no plausible draw from the sampling distribution in which the model behaves like a weak or random ranker.
        
* Thus, even if the exact point estimate of the AUC is noisy (wide interval), the regime of performance is qualitatively strong.
    

## 3 Stability of Ranking vs. Instability of Thresholded Metrics

The Validation AUC was 0.954 and the Test AUC was 0.950. These are effectively identical within sampling error, which indicates that the relative ordering of genuine vs. counterfeit cards generalizes well across splits.

However, Validation F1 at τ* was 0.95, Test F1 at τ* was 0.82, and Test F1 at τ† (oracle) was 0.86.

This gap reflects _threshold sensitivity_ not a failure of the underlying detector.

**Why AUC is stable under class imbalance:**

* AUC depends only on the relative ordering of scores $\text{AUC} = P(s(x_{\text{fake}}) > s(x_{\text{real}}))$
    
* This is independent of both class prevalence and decision thresholds
    
* Thus, if the score distributions retain roughly the same shape and overlap structure, AUC remains stable even when the class mix changes substantially, which explains why Validation and Test AUC values agreed within sampling error
    

**Why F1 is not stable under class imbalance:**

* F1 is defined as $\text{F1} = \frac{2,\text{TP}}{2,\text{TP} + \text{FP} + \text{FN}}$, where each term depends on _absolute counts_, not just rates
    
* Under a lower presence of anomalous images:
    
    * The same false positive rate yields less true positives relative to false positives
        
    * Precision decreases even if recall is unchanged: $\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}, \mathrm{Recall} = \mathrm{TPR}$.
        
    * Small absolute changes in $\mathrm{FP}$ and $\mathrm{FN}$ have a larger proportional effect.
        
* Thus, when moving from Validation to Test, τ* produces similar score separations, but the balance between $\mathrm{FP}$ and $\mathrm{FN}$ shifts, which leads to a noticeable drop in F1.
    
* We can then say that this is a property of the metric, not of the model degrading.
    

# Significance

## 1 Practical Significance for Counterfeit Detection

The results demonstrate that the model learns a compact manifold of genuine card appearance using only 200 real examples, and counterfeit cards fall outside this manifold with high probability. Importantly, detection is driven by local deviations (patch-level features), not global memorization.

AUC $\approx$ 0.95 means that ranking cards by anomaly score is highly reliable. Even though in this experiment the class imbalance led to poorer F1 scores and thus binary separation, ranking remained strong, which means that the system seems to be well-suited for _triage_, where high-score cards are set aside for manual inspection (rather than making binary decisions using the model).

Additionally, "post-hoc" thresholding can be done to match cost or prevalence assumptions, which is shown by the experiment to be plausibly viable, since the validation set showed a strong $\mathrm{F1} \approx 0.95$ when thresholding was tuned on the validation set operating regime.

## 2 Methodological Significance

This experiment did not perform a closed-set classification; no counterfeit labels were used during training. The detector thus remains valid as counterfeit styles of Pokemon cards evolve, provided that the appearance of genuine cards remains stable. This matches the true asymmetry within the problem domain, where genuine samples are abundant and fake samples are rare, heterogeneous, and continually evolving.

## 3 Extension to Numismatic Counterfeit Detection

The anomaly detection formulation explored in this project is not specific to Pokemon cards (or even other trading cards), and extends naturally to other domains with similar structural properties. A particularly close analogue is detection of counterfeit coins (numismatic counterfeit detection).

It has been observed that in online numismatic communities like Reddit, users frequently upload photographs of coins in order to solicit opinions regarding the authenticity of their coins. The judgements made by other users are often noisy, dependent on subjective expertise, limited by image quality, and are often met with quips regarding the infeasibility of assessing authenticity from a noisy photo alone. In professional settings, coin shops and collectors often rely on either expensive equipment (such as Sigma Metalytics testers, which are often $2,000 or more) or destructive/semi-destructive methods (like drilling, cutting, or chemical testing) to assess composition and authenticity. These approaches impose financial cost and physical risk to the asset, and destructive/semi-destructive methods are generally severely frowned-upon by the community.

The experimental setup in this project aligns closely with the coin-authentication problem, under the assumption that each model is trained only on one type of coin and only a single specimen is observed at a time. Similar to Pokemon cards:

* Genuine coins form a relatively stable and well-defined visual class
    
* Counterfeits are a continually evolving, heterogeneous, and open set
    
* Labeled counterfeit examples are sparse and incomplete
    
* Local visual deviations (like surface texture, lettering, edge patterns, and wear consistency) are often more informative than global appearance alone
    

Under these conditions, a normal-class-only trained anomaly detection model trained on genuine coin images could plausibly serve as a high-precision screening or triage tool. Binary decisions are not necessary to give the model practical value; even under the experiment's current results, such a system could rank coins by anomaly score. This is highly useful, because inquiries regarding coin authenticity observed on numismatic communities are often coupled with inquiries regarding whether grading the coin would be expected-value-positive (e.g. "is it worth it to send this coin in for grading?") Additionally, coin shops often must decide whether a coin merits further inspection, because expert inquiries often have limited availability relative to the number of coins processed by coin shops and may be expensive to commission. This mirrors the triage interpretation discussed previously regarding the Pokemon card results, in which strong ranking performance is more operationally meaningful than single, fixed decision thresholds.

In any case, machine learning models are rarely allowed to make independent decisions where cost assumptions for misclassifications are high (due to compliance and safety reasons), including the medical field or in counterfeit detection, so utilization of the model as a triage tool exclusively is natural irrespective of previously observed/computed F1 scores.

**Practical constraints and Image Acquisition Assumptions**

A critical assumption underlying the proposed extension is availability of high-quality and standardized images. The results in this project depend on consistent lighting, minimal background variation, and stable camera positioning. Without such conditions, significant preprocessing and normalization would be required, which may not be appropriate or reliable given the typical misclassification cost and risk assumptions in counterfeit detection.

To mitigate this, a practical deployment would likely require the following:

* A simple, low-cost physical apparatus to stabilize a smartphone or small camera
    
* Controlled lighting and fixed distance to ensure consistent scale and reflection
    
* A minimal hardware interface (e.g. a USB-connected camera or phone mount) to upload images directly for inference
    

A plausible prototyping setup is a small, enclosed 3D-printed box containing a camera and internal LED lighting. A narrow slot on the bottom allows a coin to be inserted. Once inserted, the camera captures an image, which is transferred via USB to a computer for inference using the model. The model would be integrated into a lightweight application.

This could potentially be mitigated during training for specific coins, given the availability of high-quality, stable, near-ideal images of certain coins (for example, in grading databases). However, the consideration still fully applies regarding inference.

**Positioning of the Idea**

This proposal is not presented as a fully validated solution, but as a plausible extension supported by structural similarity to the Pokemon card problem. Demonstrating feasibility for numismatic counterfeit detection would require:

* A curated dataset of genuine coins under controlled imaging conditions
    
* Evaluation across different coin types, mints, and wear levels
    
* Explicit study of robustness to lighting and viewpoint
    

However, this experiment suggests that anomaly detection may offer a low-cost and non-destructive framework for counterfeit detection in domains where genuine examples are easy to acquire and counterfeits are heterogeneous and evolving.

## Code Availability

The implementation of the full pipeline, including preprocessing, training, and evaluation, is available at:

[https://github.com/ashwin-sanjay/Fake_Pokemon_Card_Detector](https://github.com/ashwin-sanjay/Fake_Pokemon_Card_Detector)
