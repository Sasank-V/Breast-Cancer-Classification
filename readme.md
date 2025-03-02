# Breast Cancer Classification

This repository contains a Convolutional Neural Network (CNN) model to classify breast cancer images.

## Setup Instructions

Follow the steps below to set up and run the project:

### 1. Clone the Repository
```bash
git clone https://github.com/Sasank-V/Breast-Cancer-Classification.git
cd Breast-Cancer-Classification
```

### 2. Create a Virtual Environment
Create a virtual environment inside the project directory:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Train the CNN Model
Navigate to the `code` directory and run the training script:

```bash
cd code
python train.py
```

### 5. Deactivate Virtual Environment (Optional)
Once done, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure
```
Breast-Cancer-Classification/
│── code/
│   ├── train.py  # CNN training script
│── datasets/  # Dataset (not included in repo)
│   ├── all_images/ # Place the Images Data here
│── requirements.txt  # Required packages
│── README.md  # Setup and usage guide
```

## Notes
- Ensure that you have Python installed (3.8 or above recommended).
- The dataset should be placed inside the `data/` folder before training.
- Modify `train.py` as needed for hyperparameter tuning.

Happy Coding! 🚀

