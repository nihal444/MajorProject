# Mitigating Biases in Dermatological Diagnosis – GAN Augmentation & CNN Classification

*An AI-driven approach to improve fairness and accuracy in dermatological diagnosis by leveraging Generative Adversarial Networks (GANs) for dataset augmentation and Convolutional Neural Networks (CNNs) for robust classification.*

---

## 📌 Table of Contents

* <a href="#overview">Overview</a>
* <a href="#business-problem">Business Problem</a>
* <a href="#dataset">Dataset</a>
* <a href="#tools--technologies">Tools & Technologies</a>
* <a href="#data-cleaning--preparation">Data Cleaning & Preparation</a>
* <a href="#methodology">Methodology</a>
* <a href="#research-questions--key-findings">Research Questions & Key Findings</a>
* <a href="#project-images">Project Images</a>
* <a href="#how-to-run-this-project">How to Run This Project</a>
* <a href="#author--contact">Author & Contact</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>  

This project addresses **biases in dermatological diagnosis**, particularly affecting individuals with darker skin tones. Using **GAN-based augmentation** to generate synthetic lesion images and **CNN architectures (ResNet50, NASNet, InceptionResNetV2)** for classification, the study improves fairness and diagnostic accuracy.

Key features:

* Preprocessing (resizing, darkening, hair removal) for cleaner datasets
* GAN with spectral normalization to generate realistic skin lesion images
* CNN ensemble for improved classification of Melanoma, SCC, BCC, and Nevus
* Real-time prediction interface for practical deployment

---

<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>  

Dermatological datasets often lack diversity, leading to **misdiagnosis and healthcare disparities** for underrepresented populations. This project aims to:

* Mitigate dataset bias through **GAN augmentation**
* Improve classification accuracy across diverse skin tones
* Enable **real-time skin lesion detection** for medical practitioners
* Advance **healthcare equity** by reducing AI bias in medical imaging

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>  

* Source: **ISIC 2019 Skin Lesion Dataset**
* \~25,000+ dermoscopic images across multiple classes:

  * **Melanoma (MEL)**
  * **Nevus (NV)**
  * **Basal Cell Carcinoma (BCC)**
  * **Squamous Cell Carcinoma (SCC)**

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>  

* **Python** (Core language)
* **PyTorch & TensorFlow** (Deep Learning Frameworks)
* **Torch-GAN** (GAN Implementation)
* **OpenCV, PIL** (Image preprocessing & augmentation)
* **Pandas, NumPy** (Data handling)
* **Jupyter Notebook** (Experimentation & visualization)
* **GitHub** (Version control & collaboration)

---

<h2><a class="anchor" id="data-cleaning--preparation"></a>Data Cleaning & Preparation</h2>  

* **Resizing** all images to fixed dimensions (224×224 or 331×331 for NASNet)
* **Darkening augmentation** to simulate low-light conditions
* **Hair removal** using morphological operations & inpainting
* Normalization & dataset balancing with synthetic GAN images
* link to pretrained models : https://drive.google.com/drive/folders/17N_WLYjux5PYrKeZdaQePg-uo97X4V_Y?usp=sharing

---

<h2><a class="anchor" id="methodology"></a>Methodology</h2>  

1. **Preprocessing** – resize, normalize, hair removal
2. **GAN Training** – SN-GAN generates synthetic dark-skin lesion images
3. **Dataset Expansion** – combine real & synthetic samples
4. **CNN Training** – fine-tune ResNet50, NASNet, InceptionResNetV2
5. **Evaluation** – pre vs post augmentation performance
6. **Deployment** – real-time lesion prediction & UI integration

---

<h2><a class="anchor" id="research-questions--key-findings"></a>Research Questions & Key Findings</h2>  

1. **Does augmentation improve fairness?**
   ✔️ GAN augmentation improved classification across darker skin tones.

2. **Which CNN performs best?**
   ✔️ NASNet and InceptionResNetV2 achieved higher accuracy than ResNet50.

3. **Impact of preprocessing?**
   ✔️ Hair removal and darkening improved clarity and robustness.

4. **Bias reduction?**
   ✔️ Post-augmentation models showed reduced misclassification for underrepresented classes.

---

<h2><a class="anchor" id="project-images"></a>Project Images</h2>  

* Preprocessing (resizing, hair removal)
* GAN-generated synthetic lesion images
* CNN classification predictions & confusion matrices
* Training history (before & after augmentation)
* Real-time prediction UI

```markdown
![Preprocessing](images/preprocessing/sample.png)  
![GAN Outputs](images/gan_outputs/synthetic.png)  
![CNN Predictions](images/cnn_predictions/confusion_matrix.png)  
![UI Screenshot](images/ui_screenshots/app.png)  
```

---

<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>  

1. Clone the repository:

```bash
git clone https://github.com/yourusername/dermatology-bias-mitigation.git
```

2. Install dependencies:

```bash
conda create -n derm python=3.9  
conda activate derm  
pip install -r requirements.txt  
```

3. Run Jupyter notebooks for preprocessing, GAN training, and CNN classification
4. Launch the UI for real-time predictions

---

<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>  

* **Team Members:** Nihal, Kishor S Naik, Neil Mascarenhas, Prajwal P
* **Institute:** NMAM Institute of Technology, Nitte
* **Guide:** Ms. Ashwitha C Thomas (Assistant Professor, ISE Dept.)
* 📧 Contact: \[[your.nihalkanchan888@gmail.com](mailto:nihalkanchan888@gmail.com)]

