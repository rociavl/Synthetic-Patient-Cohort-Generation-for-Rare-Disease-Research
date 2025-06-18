# Synthetic Patient Cohort Generation for Rare Disease Research

**Using Conditional Generative Adversarial Networks (GANs) to Address Class Imbalance in Healthcare Data**
![image](https://github.com/user-attachments/assets/e8be8d80-c95d-4c84-a726-8c175410b23e)

## Overview

This project implements a conditional GAN architecture to generate synthetic patient data for rare disease research, specifically addressing severe class imbalance in stroke prediction datasets. The approach combines deep learning techniques with clinical domain knowledge to produce medically plausible synthetic patients while preserving statistical relationships crucial for healthcare research.

## Key Features

- **Conditional GAN Architecture**: Generates synthetic patients conditioned on stroke/non-stroke status

- **Clinical Validation**: Ensures synthetic data maintains realistic medical relationships

- **Class Imbalance Solution**: Addresses the challenge of rare disease data scarcity

- **Post-processing Enhancement**: Improves clinical realism by 77-97% across key biomarkers

- **ML Utility Testing**: Validates synthetic data effectiveness for downstream machine learning tasks

## Dataset

**Healthcare Stroke Prediction Dataset**

- **Source**: Kaggle Healthcare Dataset

- **Size**: 3,065 patient records

- **Features**: Age, hypertension, heart disease, glucose levels, BMI, work type, smoking status

- **Target**: Stroke occurrence (binary classification)

- **Class Distribution**: Highly imbalanced (minority class: stroke patients)

## Methodology

### 1. Conditional GAN Architecture

- **Generator**: Creates synthetic patient profiles conditioned on stroke status

- **Discriminator**: Distinguishes between real and synthetic patients

- **Training**: 3,000 epochs with careful hyperparameter tuning

### 2. Clinical Enhancement

- **Age Relationship**: Ensures stroke patients are older on average

- **Comorbidity Patterns**: Maintains realistic hypertension and heart disease rates

- **Glucose Levels**: Preserves diabetic risk factor relationships

### 3. Validation Framework

- **Statistical Validation**: Distribution matching and feature correlation analysis

- **Clinical Validation**: Expert-guided post-processing for medical plausibility

- **ML Utility Testing**: Performance evaluation on downstream classification tasks

## Results

### Synthetic Data Quality

- **Age Distribution**: Accurately captures age-stroke relationship

- **Hypertension**: 77% improvement in clinical realism

- **Heart Disease**: 83.7% improvement in clinical realism

- **Glucose Levels**: 97.6% improvement in clinical realism

### Machine Learning Performance

| Model | Accuracy | ROC-AUC | Data Size |
|-------|----------|---------|-----------|
| Real Only | 0.951 | 0.791 | 3,065 |
| Synthetic Only | 0.950 | 0.593 | 1,000 |
| Real + Synthetic | 0.950 | 0.782 | 4,065 |

## Installation

```bash
  # Clone the repository
  git clone https://github.com/yourusername/synthetic-patient-cohort.git
  cd synthetic-patient-cohort
  
  # Install required packages
  pip install -r requirements.txt
  
  # Install additional dependencies
  pip install kagglehub torch matplotlib pandas scikit-learn numpy
  ```

## Usage

### Basic Usage

```python
# Load and preprocess data
from src.data_loader import load_stroke_data
from src.preprocessing import preprocess_data

data = load_stroke_data()
processed_data = preprocess_data(data)

# Train conditional GAN
from src.models import ConditionalGAN

gan = ConditionalGAN()
gan.train(processed_data, epochs=3000)

# Generate synthetic patients
synthetic_stroke = gan.generate_patients(condition=1, n_samples=1000)
synthetic_no_stroke = gan.generate_patients(condition=0, n_samples=1000)

# Apply clinical enhancement
from src.enhancement import enhance_clinical_realism

enhanced_stroke = enhance_clinical_realism(synthetic_stroke, target_condition=1)
```

### Advanced Usage

```python
# Custom training configuration
config = {
    'learning_rate': 0.0002,
    'batch_size': 64,
    'latent_dim': 100,
    'epochs': 3000
}

gan = ConditionalGAN(config)
gan.train(processed_data)

# Comprehensive evaluation
from src.evaluation import evaluate_synthetic_data

results = evaluate_synthetic_data(
    real_data=processed_data,
    synthetic_data=enhanced_stroke,
    metrics=['statistical', 'clinical', 'ml_utility']
)
```

## Project Structure

```
synthetic-patient-cohort/
├── README.md
├── requirements.txt
├── LICENSE
├── notebooks/
│   └── Synthetic_Patient_Cohort_Generation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gan.py
│   │   └── discriminator.py
│   ├── enhancement.py
│   ├── evaluation.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── models/
│   ├── figures/
│   └── synthetic_data/
└── tests/
    ├── test_models.py
    ├── test_enhancement.py
    └── test_evaluation.py
```

## Requirements

- Python 3.8+

- PyTorch 1.9+

- pandas

- numpy

- scikit-learn

- matplotlib

- seaborn

- kagglehub

## Ethics and Privacy

This project generates synthetic medical data to address privacy concerns while enabling rare disease research. Key considerations:

- **Privacy Protection**: No real patient data is exposed or shared

- **Ethical Use**: Synthetic data should only be used for research purposes

- **Medical Validation**: Clinical expertise should validate synthetic data before use

- **Regulatory Compliance**: Consider HIPAA, GDPR, and other relevant regulations

## Limitations

- **Domain Specificity**: Trained specifically on stroke prediction data

- **Feature Set**: Limited to available features in the source dataset

- **Clinical Validation**: Requires expert medical review for real-world applications

- **Generalization**: May not generalize to other rare diseases without retraining

## Contributing

We welcome contributions to improve synthetic data generation for healthcare research:

1. Fork the repository

2. Create a feature branch (`git checkout -b feature/improvement`)

3. Commit your changes (`git commit -am 'Add new feature'`)

4. Push to the branch (`git push origin feature/improvement`)

5. Create a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{avalos2025synthetic,
  title={Synthetic Patient Cohort Generation for Rare Disease Research Using Conditional GANs},
  author={Ávalos, Rocío},
  institution={UPC},
  year={2025},
  month={May}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Rocío Ávalos**  

Institution: UPC  

Project Date: May 2025

For questions, suggestions, or collaborations, please open an issue or contact the author.

---

**Disclaimer**: This synthetic data generation tool is designed for research purposes only. Always consult with medical professionals and follow appropriate ethical guidelines when working with healthcare data, even synthetic data.
