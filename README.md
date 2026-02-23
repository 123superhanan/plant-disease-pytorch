

```
# Plant Disease Detection with PyTorch

This project classifies plant diseases using images from the PlantVillage dataset.

It uses PyTorch and a convolutional neural network for recognition.

## Features

- Loads and explores PlantVillage dataset (20,638 images, 15 classes)
- Handles class imbalance analysis
- Trains image classification model
- Shows accuracy on validation set

## Requirements

- Python 3.8+
- PyTorch & Torchvision
- Pandas, Matplotlib, Scikit-learn

Install everything with:

```bash
pip install torch torchvision torchaudio pandas matplotlib scikit-learn
```

## Setup

1. Clone the repo
```bash
git clone https://github.com/yourusername/plant-disease-pytorch.git
cd plant-disease-pytorch
```

2. Create and activate virtual environment
```bash
python -m venv torch_env
torch_env\Scripts\activate   # Windows
# or source torch_env/bin/activate   # Mac/Linux
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

(If you do not have requirements.txt yet, run `pip freeze > requirements.txt` after installs)

4. Run the notebook
```bash
jupyter notebook pytorch.ipynb
```

## Dataset

PlantVillage dataset with 20,638 images across 15 disease/healthy classes.

Classes include Tomato, Pepper, Potato diseases.

Dataset is local (not in repo due to size).

Download and extract to PlantVillage/ folder.

## Notebook Structure

- Data exploration with pandas
- Class imbalance visualization
- Train/val split
- Custom Dataset class
- Model training (EfficientNet or ResNet)
- Evaluation

## Results

Validation accuracy depends on training.

Typical range: 85-95% with fine-tuning.

## License


```

