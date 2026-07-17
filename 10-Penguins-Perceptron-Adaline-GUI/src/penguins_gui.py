#!/usr/bin/env python3
"""
penguins_gui.py
Tkinter GUI for Perceptron and Adaline on penguins.csv

Usage:
    python penguins_gui.py

Requires: pandas, numpy, matplotlib
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

np.random.seed(42)

# -------------------------
# Utilities (data & metrics)
# -------------------------
def impute_missing(df):
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
        else:
            if df[col].isnull().any():
                m = df[col].mode()
                if len(m) > 0:
                    df[col].fillna(m[0], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)
    return df

def encode_origin(df):
    if 'originlocation' in df.columns:
        vals = sorted(df['originlocation'].unique())
        mapping = {v: i for i, v in enumerate(vals)}
        df['originlocation'] = df['originlocation'].map(mapping)
        return mapping
    return None

def standardize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma

def compute_confusion(y_true, y_pred, pos_label=1, neg_label=-1):
    TP = int(np.sum((y_true == pos_label) & (y_pred == pos_label)))
    TN = int(np.sum((y_true == neg_label) & (y_pred == neg_label)))
    FP = int(np.sum((y_true == neg_label) & (y_pred == pos_label)))
    FN = int(np.sum((y_true == pos_label) & (y_pred == neg_label)))
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0.0
    return {'TP':TP,'TN':TN,'FP':FP,'FN':FN,'accuracy':accuracy}

# -------------------------
# Perceptron & Adaline
# -------------------------
class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, n_epochs=50, add_bias=True):
        self.lr = learning_rate
        self.epochs = n_epochs
        self.add_bias = add_bias
        self.w = None
        self.b = 0.0

    def net_input(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        net = self.net_input(X)
        return np.where(net >= 0.0, 1, -1)

    def fit(self, X, y, verbose=False):
        n_samples, n_features = X.shape
        self.w = np.random.normal(scale=0.01, size=n_features)
        if self.add_bias:
            self.b = 0.0
        errors_per_epoch = []
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                net = np.dot(xi, self.w) + self.b
                y_pred = 1 if net >= 0 else -1
                update = self.lr * (target - y_pred)
                if update != 0.0:
                    self.w += update * xi
                    if self.add_bias:
                        self.b += update
                    errors += 1
            errors_per_epoch.append(errors)
            if verbose:
                print(f"[Perceptron] Epoch {epoch+1}/{self.epochs}, updates={errors}")
            if errors == 0:
                break
        return errors_per_epoch

class Adaline:
    def __init__(self, n_features, learning_rate=0.01, n_epochs=50, add_bias=True):
        self.lr = learning_rate
        self.epochs = n_epochs
        self.add_bias = add_bias
        self.w = None
        self.b = 0.0

    def net_input(self, X):
        return X.dot(self.w) + self.b

    def predict(self, X):
        net = self.net_input(X)
        return np.where(net >= 0.0, 1, -1)

    def fit(self, X, y, mse_threshold=1e-4, verbose=False):
        n_samples, n_features = X.shape
        self.w = np.random.normal(scale=0.01, size=n_features)
        if self.add_bias:
            self.b = 0.0
        mse_per_epoch = []
        for epoch in range(self.epochs):
            net = self.net_input(X)
            error = y - net
            self.w += self.lr * X.T.dot(error) / n_samples
            if self.add_bias:
                self.b += self.lr * error.mean()
            mse = (error**2).mean()
            mse_per_epoch.append(mse)
            if verbose:
                print(f"[Adaline] Epoch {epoch+1}/{self.epochs}, MSE={mse:.6f}")
            if mse < mse_threshold:
                break
        return mse_per_epoch

# -------------------------
# Plotting (separate window)
# -------------------------
def plot_decision_boundary(X, y, model, feature_names, title):
    # X is standardized Nx2
    plt.figure(figsize=(7,5))
    # scatter points
    for lab, marker in zip([1, -1], ('o', 's')):
        ix = np.where(y == lab)
        plt.scatter(X[ix,0], X[ix,1], marker=marker, label=f'class {lab}', alpha=0.8)
    # decision boundary
    w = model.w
    b = model.b
    if abs(w[1]) < 1e-8:
        x_line = -b / w[0]
        plt.axvline(x=x_line, color='red', linestyle='--', label='decision boundary')
    else:
        xx = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
        yy = -(w[0]/w[1]) * xx - (b / w[1])
        plt.plot(xx, yy, 'r--', label='decision boundary')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show(block=False)

def plot_confusion_matrix(conf, title="Confusion Matrix"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,4))
    mat = [[conf['TP'], conf['FN']],
           [conf['FP'], conf['TN']]]
    ax.imshow(mat, cmap='Blues')

    # axis labels
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred +1','Pred -1'])
    ax.set_yticklabels(['Actual +1','Actual -1'])
    ax.set_title(title)

    # write values inside cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i][j]),
                    ha='center', va='center', color='black', fontsize=14, fontweight='bold')

    # display overall accuracy below the plot
    acc = conf['accuracy'] * 100
    ax.text(0.5, 1.12, f'Accuracy: {acc:.2f}%', transform=ax.transAxes,
            ha='center', fontsize=12, color='darkblue')

    plt.tight_layout()
    plt.show(block=False)

# -------------------------
# GUI
# -------------------------
class PenguinApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Penguins Classifier (Perceptron / Adaline)")
        self.geometry("760x520")
        self.resizable(False, False)

        # Try load CSV
        self.csv_path = os.getenv("PENGUINS_CSV", "data/penguins.csv")
        if not os.path.exists(self.csv_path):
            messagebox.showerror("File error", f"File '{self.csv_path}' not found in current folder.")
            self.destroy()
            return

        # read and preprocess header (lowercase columns)
        df = pd.read_csv(self.csv_path)
        df.columns = [c.lower() for c in df.columns]
        df = impute_missing(df)
        encode_origin(df)
        self.df = df

        # GUI elements
        self.create_widgets()
        self.populate_options()

    def create_widgets(self):
        padx = 8
        pady = 6

        # Frame left: controls
        left = ttk.Frame(self, padding=(10,10))
        left.place(x=10, y=10, width=360, height=500)

        ttk.Label(left, text="Select first class (A):").grid(row=0, column=0, sticky="w", padx=padx, pady=pady)
        self.cls1_cb = ttk.Combobox(left, state="readonly")
        self.cls1_cb.grid(row=1, column=0, sticky="we", padx=padx)

        ttk.Label(left, text="Select second class (B):").grid(row=2, column=0, sticky="w", padx=padx, pady=pady)
        self.cls2_cb = ttk.Combobox(left, state="readonly")
        self.cls2_cb.grid(row=3, column=0, sticky="we", padx=padx)

        ttk.Separator(left, orient='horizontal').grid(row=4, column=0, sticky="we", pady=8)

        ttk.Label(left, text="Feature 1:").grid(row=5, column=0, sticky="w", padx=padx)
        self.feat1_cb = ttk.Combobox(left, state="readonly")
        self.feat1_cb.grid(row=6, column=0, sticky="we", padx=padx)

        ttk.Label(left, text="Feature 2:").grid(row=7, column=0, sticky="w", padx=padx)
        self.feat2_cb = ttk.Combobox(left, state="readonly")
        self.feat2_cb.grid(row=8, column=0, sticky="we", padx=padx)

        ttk.Separator(left, orient='horizontal').grid(row=9, column=0, sticky="we", pady=8)

        # Hyperparams
        ttk.Label(left, text="Learning rate (eta):").grid(row=10, column=0, sticky="w", padx=padx)
        self.lr_entry = ttk.Entry(left); self.lr_entry.insert(0, "0.01")
        self.lr_entry.grid(row=11, column=0, sticky="we", padx=padx)

        ttk.Label(left, text="Epochs:").grid(row=12, column=0, sticky="w", padx=padx)
        self.epochs_entry = ttk.Entry(left); self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=13, column=0, sticky="we", padx=padx)

        ttk.Label(left, text="MSE threshold (Adaline):").grid(row=14, column=0, sticky="w", padx=padx)
        self.mse_entry = ttk.Entry(left); self.mse_entry.insert(0, "1e-4")
        self.mse_entry.grid(row=15, column=0, sticky="we", padx=padx)

        self.bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Add bias", variable=self.bias_var).grid(row=16, column=0, sticky="w", padx=padx, pady=6)

        ttk.Label(left, text="Algorithm:").grid(row=17, column=0, sticky="w", padx=padx)
        self.algo_cb = ttk.Combobox(left, state="readonly", values=["Perceptron", "Adaline", "Both"])
        self.algo_cb.current(2)
        self.algo_cb.grid(row=18, column=0, sticky="we", padx=padx, pady=(0,8))

        ttk.Button(left, text="Train & Plot", command=self.on_train).grid(row=19, column=0, sticky="we", padx=padx, pady=(6,2))
        ttk.Button(left, text="Quit", command=self.destroy).grid(row=20, column=0, sticky="we", padx=padx, pady=(2,6))

        # Frame right: results log
        right = ttk.Frame(self, padding=(10,10))
        right.place(x=380, y=10, width=370, height=500)

        ttk.Label(right, text="Status / Results:").pack(anchor="w")
        self.log = scrolledtext.ScrolledText(right, width=44, height=28, state='normal')
        self.log.pack(fill='both', expand=True, pady=(6,0))

    def populate_options(self):
        species_list = sorted(self.df['species'].unique())
        self.cls1_cb['values'] = species_list
        self.cls2_cb['values'] = species_list
        # make defaults
        if len(species_list) >= 2:
            self.cls1_cb.current(0)
            self.cls2_cb.current(1)

        features = [c for c in self.df.columns if c != 'species']
        self.feat1_cb['values'] = features
        self.feat2_cb['values'] = features
        if len(features) >= 2:
            self.feat1_cb.current(0)
            self.feat2_cb.current(1)

    def append_log(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def on_train(self):
        try:
            class1 = self.cls1_cb.get()
            class2 = self.cls2_cb.get()
            if not class1 or not class2 or class1 == class2:
                messagebox.showwarning("Invalid classes", "Select two different classes.")
                return
            f1 = self.feat1_cb.get()
            f2 = self.feat2_cb.get()
            if f1 == f2:
                messagebox.showwarning("Invalid features", "Pick two different features.")
                return

            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())
            mse_th = float(self.mse_entry.get())
            add_bias = bool(self.bias_var.get())
            algo = self.algo_cb.get()

            # prepare data
            df2 = self.df[self.df['species'].isin([class1, class2])].copy()
            X_all = df2[[f1, f2]].values.astype(float)
            X_all_std, mu, sigma = standardize(X_all)
            species_to_label = {class1: 1, class2: -1}
            y_all = df2['species'].map(species_to_label).values

            # shuffle per class and split (30 train, 20 test per class if possible)
            idxs = []
            for cls in [class1, class2]:
                cls_idx = np.where(df2['species'].values == cls)[0]
                np.random.shuffle(cls_idx)
                idxs.append(cls_idx)
            idxs1, idxs2 = idxs
            n_take = min(50, len(idxs1), len(idxs2))
            train_per_class = min(30, n_take)
            test_per_class = min(20, n_take - train_per_class)
            if test_per_class <= 0:
                messagebox.showwarning("Not enough samples", "Not enough samples for 30 train + 20 test split; code will use available samples.")
                test_per_class = max(0, n_take - train_per_class)

            train_idx = np.concatenate([idxs1[:train_per_class], idxs2[:train_per_class]])
            test_idx = np.concatenate([idxs1[train_per_class:train_per_class+test_per_class],
                                      idxs2[train_per_class:train_per_class+test_per_class]])

            X_train = X_all_std[train_idx]
            y_train = y_all[train_idx]
            X_test = X_all_std[test_idx]
            y_test = y_all[test_idx]

            self.append_log(f"Selected classes: {class1} vs {class2}")
            self.append_log(f"Selected features: {f1}, {f2}")
            self.append_log(f"Train per class: {train_per_class} | Test per class: {test_per_class}")
            self.append_log(f"Learning rate: {lr}, Epochs: {epochs}, MSE th: {mse_th}, Add bias: {add_bias}")
            self.append_log(f"Algorithm: {algo}")

            results = {}

            if algo in ("Perceptron", "Both"):
                self.append_log("\n--- Training Perceptron ---")
                pmodel = Perceptron(n_features=X_train.shape[1], learning_rate=lr, n_epochs=epochs, add_bias=add_bias)
                errs = pmodel.fit(X_train, y_train, verbose=False)
                y_pred_test = pmodel.predict(X_test)
                conf = compute_confusion(y_test, y_pred_test, pos_label=1, neg_label=-1)
                results['perceptron'] = {'model':pmodel, 'conf':conf}
                self.append_log(f"Perceptron confusion: {conf}")
                # plot
                plot_decision_boundary(X_test, y_test, pmodel, [f1, f2], title=f"Perceptron: {class1} vs {class2}")
                # ---  الإضافة الأولى هنا  ---
                plot_confusion_matrix(conf, title=f"Perceptron Confusion: {class1} vs {class2}")


            if algo in ("Adaline", "Both"):
                self.append_log("\n--- Training Adaline ---")
                amodel = Adaline(n_features=X_train.shape[1], learning_rate=lr, n_epochs=epochs, add_bias=add_bias)
                mses = amodel.fit(X_train, y_train, mse_threshold=mse_th, verbose=False)
                y_pred_test = amodel.predict(X_test)
                conf = compute_confusion(y_test, y_pred_test, pos_label=1, neg_label=-1)
                results['adaline'] = {'model':amodel, 'conf':conf}
                self.append_log(f"Adaline confusion: {conf}")
                plot_decision_boundary(X_test, y_test, amodel, [f1, f2], title=f"Adaline: {class1} vs {class2}")
                # ---  الإضافة الثانية هنا  ---
                plot_confusion_matrix(conf, title=f"Adaline Confusion: {class1} vs {class2}")

            # summary
            self.append_log("\n=== Summary ===")
            for k, v in results.items():
                c = v['conf']
                self.append_log(f"{k.upper()}: Accuracy = {c['accuracy']*100:.2f}% | TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}")

            self.append_log("\nDone.\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            raise


if __name__ == "__main__":
    app = PenguinApp()
    app.mainloop()