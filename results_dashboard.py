import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Paths to metrics and images
metrics_dir = "metrics"
accuracy_file = os.path.join(metrics_dir, "accuracy.txt")
sentiment_distribution_image = os.path.join(metrics_dir, "sentiment_distribution.png")

# Read accuracy metrics
metrics = []
if os.path.exists(accuracy_file):
    with open(accuracy_file, "r") as file:
        metrics = file.readlines()
else:
    messagebox.showerror("Error", f"Accuracy file not found at {accuracy_file}")
    exit()

# Create the main window
root = tk.Tk()
root.title("Sentiment Analysis Results")
root.geometry("800x600")

# Create a notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Tab 1: Metrics Tab
metrics_tab = ttk.Frame(notebook)
notebook.add(metrics_tab, text="Metrics")

# Tab 2: Sentiment Distribution Tab
sentiment_tab = ttk.Frame(notebook)
notebook.add(sentiment_tab, text="Sentiment Distribution")

# Load and display the sentiment distribution image in the Sentiment Distribution Tab
if os.path.exists(sentiment_distribution_image):
    try:
        sentiment_image = Image.open(sentiment_distribution_image)
        sentiment_image = sentiment_image.resize((400, 300), Image.Resampling.LANCZOS)
        sentiment_image_tk = ImageTk.PhotoImage(sentiment_image)

        sentiment_label = tk.Label(sentiment_tab, image=sentiment_image_tk)
        sentiment_label.image = sentiment_image_tk
        sentiment_label.pack(pady=10)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load sentiment image: {e}")
else:
    messagebox.showerror("Error", f"Sentiment distribution image not found at {sentiment_distribution_image}")

# Metrics Tab: Create a figure for the bar chart and numerical chart
combined_fig, metrics_ax = plt.subplots(figsize=(8, 5))

# Extract metrics data for plotting
model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

# Debugging: Ensure metrics data is parsed correctly
try:
    for line in metrics:
        if "Accuracy" in line:
            model_names.append(line.split()[0])
            accuracies.append(float(line.split()[-1]))
        elif "Precision" in line:
            precisions.append(float(line.split()[-1]))
        elif "Recall" in line:
            recalls.append(float(line.split()[-1]))
        elif "F1-Score" in line:
            f1_scores.append(float(line.split()[-1]))
except ValueError as e:
    messagebox.showerror("Error", f"Failed to parse metrics: {e}")
    exit()

# Debugging: Print parsed metrics to verify
print("Parsed Metrics:")
print("Model Names:", model_names)
print("Accuracies:", accuracies)
print("Precisions:", precisions)
print("Recalls:", recalls)
print("F1-Scores:", f1_scores)

# Plot metrics (bar chart)
x = range(len(model_names))
metrics_ax.bar(x, accuracies, width=0.2, label="Accuracy", align="center")
metrics_ax.bar([i + 0.2 for i in x], precisions, width=0.2, label="Precision", align="center")
metrics_ax.bar([i + 0.4 for i in x], recalls, width=0.2, label="Recall", align="center")
metrics_ax.bar([i + 0.6 for i in x], f1_scores, width=0.2, label="F1-Score", align="center")

metrics_ax.set_xticks([i + 0.3 for i in x])
metrics_ax.set_xticklabels(model_names)
metrics_ax.set_title("Model Metrics Comparison")
metrics_ax.set_ylabel("Scores")
metrics_ax.legend()

# Add a table below the metrics chart
columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
cell_text = [
    [model, f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}"]
    for model, acc, prec, rec, f1 in zip(model_names, accuracies, precisions, recalls, f1_scores)
]
metrics_ax.table(
    cellText=cell_text,
    colLabels=columns,
    cellLoc="center",
    loc="bottom",
    bbox=[0.0, -0.5, 1.0, 0.3]
)
metrics_ax.figure.subplots_adjust(bottom=0.4)

# Add the combined figure to the Metrics tab
canvas_metrics = FigureCanvasTkAgg(combined_fig, master=metrics_tab)
canvas_metrics.get_tk_widget().pack(fill="both", expand=True)

# Run the application
root.mainloop()
