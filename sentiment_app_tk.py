# import tkinter as tk
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np

# # Load model
# model_path = "./final_model_cpu"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)

# def predict():
#     text = entry.get("1.0", tk.END).strip()
#     if not text:
#         result_label.config(text="Please enter some text")
#         return
#     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
#     pred = np.argmax(probs)
#     labels = ["negative", "positive"]
#     result_label.config(text=f"Prediction: {labels[pred]} ({probs[pred]*100:.2f}% confidence)")

# # Tkinter UI
# root = tk.Tk()
# root.title("Sentiment Analysis App")

# tk.Label(root, text="Enter review:").pack()
# entry = tk.Text(root, height=5, width=50)
# entry.pack()

# tk.Button(root, text="Predict", command=predict).pack(pady=5)

# result_label = tk.Label(root, text="")
# result_label.pack()

# root.mainloop()






import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model
model_path = "./final_model_cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["negative", "positive"]

# ---- Single Prediction ----
def predict():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        result_label.config(text="Please enter some text")
        return

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = np.argmax(probs)
    result_label.config(
        text=f"Prediction: {labels[pred]} ({probs[pred]*100:.2f}% confidence)"
    )

# ---- Batch Prediction ----
def batch_predict():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path)
        if "review" not in df.columns:
            messagebox.showerror("Error", "CSV must contain a 'review' column.")
            return

        predictions = []
        confidences = []

        for text in df["review"].astype(str).tolist():
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)
            predictions.append(labels[pred])
            confidences.append(float(probs[pred]))

        df["predicted_sentiment"] = predictions
        df["confidence"] = confidences

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"Predictions saved to {save_path}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ---- Tkinter UI ----
root = tk.Tk()
root.title("Sentiment Analysis App")

tk.Label(root, text="Enter review:").pack()
entry = tk.Text(root, height=5, width=50)
entry.pack()

tk.Button(root, text="Predict", command=predict).pack(pady=5)
tk.Button(root, text="Batch Predict (CSV)", command=batch_predict).pack(pady=5)

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
