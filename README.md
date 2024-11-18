_(A learning project)_



# Named Entity Recognition with BERT: Fine-Tuning on CoNLL-2003 Dataset

**Author:** Felipe Rodriguez Angel
**Video:** [https://youtu.be/phO7AJKw-9E](https://youtu.be/phO7AJKw-9E)
---

## 📖 **Project Overview**

This project focuses on building a high-performance Named Entity Recognition (NER) model using **transfer learning** with a pre-trained **BERT** model. The model is fine-tuned on the **CoNLL-2003 dataset**, a widely recognized benchmark dataset for NER tasks. The primary goal is to accurately identify and categorize named entities in text, such as:

- **Person (PER):** Names of individuals (e.g., "John Doe").
- **Organization (ORG):** Names of companies or institutions (e.g., "Google").
- **Location (LOC):** Names of geographical locations (e.g., "New York").
- **Miscellaneous (MISC):** Other entities that do not fit the above categories (e.g., "Olympics").

This project demonstrates the power of transfer learning in NLP, leveraging BERT's pre-trained language model to effectively solve a sequence labeling task.

---

## 💡 **Key Features**

- **Data Exploration:** An in-depth analysis of the CoNLL-2003 dataset, including class distribution and sentence length statistics.
- **Data Preprocessing:** Tokenization using BERT's tokenizer, dynamic padding, and class weighting to handle class imbalance.
- **Model Training:** Fine-tuning of a BERT-based model with mixed-precision training for efficient use of computational resources.
- **Evaluation:** Comprehensive analysis using metrics such as precision, recall, F1-score, and confusion matrix visualizations.

---

## 📚 **Dataset**

The model is trained and evaluated on the **CoNLL-2003 dataset**, which is publicly available on Hugging Face. This dataset is a benchmark for Named Entity Recognition and contains labeled examples for four entity types (PER, ORG, LOC, MISC). The dataset can be accessed using the Hugging Face Datasets library:

🔗 [CoNLL-2003 Dataset on Hugging Face](https://huggingface.co/datasets/conll2003)

To load the dataset in your own project, use:

```python
from datasets import load_dataset

dataset = load_dataset("conll2003")
```

---

## 📁 **Project Structure**

```
├── 01_data_exploration.ipynb         # Data exploration and analysis
├── 02_data_preprocessing_feature_engineering.ipynb  # Data preprocessing and feature engineering
├── 03_model_training_and_evaluation.ipynb  # Model training and evaluation
├── README.md                         # Project overview and instructions
├── results/                          # Folder containing model results and visualizations
```

---

## 🔍 **How to Use**

1. Clone the repository and open the Jupyter notebooks in Google Colab for an easy, reproducible environment.
2. Run `01_data_exploration.ipynb` to explore the dataset and gain insights into the data distribution.
3. Proceed to `02_data_preprocessing_feature_engineering.ipynb` for data preparation and feature engineering.
4. Finally, open `03_model_training_and_evaluation.ipynb` to fine-tune the BERT model and evaluate its performance.

---

## 🛠️ **Dependencies**

- Python 3.10+
- `transformers` library (Hugging Face)
- `datasets` library (Hugging Face)
- `scikit-learn` for evaluation metrics
- `matplotlib` and `seaborn` for visualizations

Install the required packages with:

```bash
pip install transformers datasets scikit-learn matplotlib seaborn
```

---

## 📈 **Results**

The fine-tuned BERT model achieved strong performance on both validation and test sets, with high precision, recall, and F1-scores across all entity types. The results and visualizations can be found in the `03_model_training_and_evaluation.ipynb` notebook.

---

## 📋 **References**

1. **Sang, E. F. T. K., & De Meulder, F.** (2003). Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. In *Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003* (pp. 142-147).
2. **Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K.** (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)* (pp. 4171-4186).
3. **Marr, B.** (2021). How Named Entity Recognition Can Transform Legal Tech and Document Management. *Forbes Technology Council*.
4. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). Attention is All You Need. In *Advances in Neural Information Processing Systems* (pp. 6000-6010).

---
