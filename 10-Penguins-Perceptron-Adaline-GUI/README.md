# Penguins Perceptron Adaline GUI

A Python Tkinter desktop app for training and comparing Perceptron and Adaline classifiers on a penguins dataset.

## Features

- Loads an included penguins CSV dataset.
- Handles missing values.
- Encodes categorical origin/location values.
- Standardizes selected numeric features.
- Trains custom Perceptron and Adaline implementations.
- Shows model accuracy and confusion-matrix style metrics.
- Plots decision boundaries and training behavior with Matplotlib.

## Tech Stack

- Python
- Tkinter
- NumPy
- pandas
- Matplotlib

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run from the project root so the app can find `data/penguins.csv`:

```powershell
python src\penguins_gui.py
```

You can also point to a different CSV:

```powershell
$env:PENGUINS_CSV="D:\path\to\penguins.csv"
python src\penguins_gui.py
```

## Dataset

The included dataset contains penguin species, bill measurements, flipper length, origin location, and body mass.

## GitHub Readiness

Status: good small ML app candidate.

Recommended before publishing:

- Add a screenshot of the GUI.
- Add a short explanation of Perceptron vs Adaline.
- Add a license if public.
- Optional: add unit tests for preprocessing and model classes.
