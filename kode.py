"""
Crop Failure / Fire Risk Prediction Dashboard with LSTM Model
Dependencies:
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- joblib
- tkinter (standard)

Make sure you install TensorFlow before running:
  pip install tensorflow
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Main colors
BG_COLOR = '#688b9a'
ELEMENT_COLOR = '#47697E'
TEXT_COLOR = 'white'


class DashboardFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)
        
        # Main container frame
        main_container = tk.Frame(self, bg=BG_COLOR)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Dashboard controls and chart
        left_frame = tk.Frame(main_container, bg=BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side - Manual input form
        right_frame = tk.Frame(main_container, bg=BG_COLOR, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Dashboard title and controls
        tk.Label(left_frame, text="Dashboard Prediksi Kebakaran", 
                font=('Times New Roman', 16, 'bold'), bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=10)

        control_frame = tk.Frame(left_frame, bg=BG_COLOR)
        control_frame.pack(pady=5)

        self.periode_var = tk.StringVar(value='Harian')
        ttk.Label(control_frame, text="Pilih Periode:", font=('Times New Roman', 12), 
                 background=BG_COLOR, foreground=TEXT_COLOR).grid(row=0, column=0, sticky='w')
        self.periode_combo = ttk.Combobox(control_frame, textvariable=self.periode_var,
                                        values=["Harian", "Mingguan", "Bulanan"],
                                        font=('Times New Roman', 12), state="readonly", width=15)
        self.periode_combo.grid(row=0, column=1, padx=5)
        self.periode_combo.bind("<<ComboboxSelected>>", lambda e: self.toggle_custom_entry())

        self.custom_entry = tk.Entry(control_frame, font=('Times New Roman', 12), width=5, bg=ELEMENT_COLOR, fg=TEXT_COLOR)
        self.custom_entry.insert(0, "8")
        self.custom_entry.grid(row=0, column=2, padx=5)
        self.custom_entry.grid_remove()

        self.btn_show = tk.Button(control_frame, text="Tampilkan Grafik", 
                                font=('Times New Roman', 12), command=self.update_chart,
                                bg=ELEMENT_COLOR, fg=TEXT_COLOR, activebackground=ELEMENT_COLOR, activeforeground=TEXT_COLOR)
        self.btn_show.grid(row=1, columnspan=3, pady=10)

        # Chart area
        self.figure = Figure(figsize=(6, 3.5), dpi=100, facecolor=BG_COLOR)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BG_COLOR)

        self.canvas = FigureCanvasTkAgg(self.figure, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Manual input form in right frame
        tk.Label(right_frame, text="Input Data Manual", 
                font=('Times New Roman', 16, 'bold'), bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=10)

        form_frame = tk.Frame(right_frame, bg=BG_COLOR)
        form_frame.pack(pady=5)

        labels = [
            "Waktu (YYYY-MM-DD HH:MM):", 
            "Suhu (°C):", 
            "Asap (ppm):", 
            "Gas (ppm):"  # Gas for leak detection
        ]

        self.entries = {}
        for i, label_text in enumerate(labels):
            tk.Label(form_frame, text=label_text, font=('Times New Roman', 10), 
                    bg=BG_COLOR, fg=TEXT_COLOR).grid(row=i, column=0, sticky='e', pady=2)
            entry = tk.Entry(form_frame, font=('Times New Roman', 10), width=15, bg=ELEMENT_COLOR, fg=TEXT_COLOR)
            entry.grid(row=i, column=1, pady=2, padx=5)
            self.entries[label_text] = entry

        # Set default values
        self.entries["Waktu (YYYY-MM-DD HH:MM):"].insert(0, "2023-01-01 08:00")
        self.entries["Suhu (°C):"].insert(0, "25")
        self.entries["Asap (ppm):"].insert(0, "20")
        self.entries["Gas (ppm):"].insert(0, "5")

        self.result_label = tk.Label(right_frame, text="", 
                                    font=('Times New Roman', 12, 'bold'), 
                                    fg=TEXT_COLOR, bg=BG_COLOR, wraplength=250)
        self.result_label.pack(pady=5)

        # Button frame for Predict and Save buttons
        button_frame = tk.Frame(right_frame, bg=BG_COLOR)
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="Prediksi", font=('Times New Roman', 12), 
                 command=self.predict_manual, bg=ELEMENT_COLOR, fg=TEXT_COLOR,
                 activebackground=ELEMENT_COLOR, activeforeground=TEXT_COLOR).pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Simpan Data", font=('Times New Roman', 12), 
                 command=self.save_manual_data, bg=ELEMENT_COLOR, fg=TEXT_COLOR,
                 activebackground=ELEMENT_COLOR, activeforeground=TEXT_COLOR).pack(side=tk.LEFT, padx=5)

    def save_manual_data(self):
        try:
            data = {
                'Waktu': self.entries["Waktu (YYYY-MM-DD HH:MM):"].get(),
                'Suhu (°C)': float(self.entries["Suhu (°C):"].get()),
                'Asap (ppm)': float(self.entries["Asap (ppm):"].get()),
                'Gas (ppm)': float(self.entries["Gas (ppm):"].get())
            }
            df = pd.DataFrame([data])
            if "Hasil Prediksi:" in self.result_label.cget("text"):
                prediksi = self.result_label.cget("text").split(": ")[1]
                df['Status'] = prediksi
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
                title="Simpan Data Input Manual"
            )

            if file_path:
                try:
                    existing_df = pd.read_excel(file_path)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_excel(file_path, index=False)
                except:
                    df.to_excel(file_path, index=False)
                messagebox.showinfo("Sukses", f"Data berhasil disimpan di:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan data:\n{e}")

    def predict_manual(self):
        try:
            waktu = self.entries["Waktu (YYYY-MM-DD HH:MM):"].get()
            suhu = float(self.entries["Suhu (°C):"].get())
            asap = float(self.entries["Asap (ppm):"].get())
            gas = float(self.entries["Gas (ppm):"].get())

            pd.to_datetime(waktu, format='%Y-%m-%d %H:%M', errors='raise')

            # Load model and scaler if available
            model_path = 'fire_detection_lstm.h5'
            scaler_path = 'feature_scaler.pkl'
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    model = load_model(model_path)
                    scaler = joblib.load(scaler_path)
                    
                    features = np.array([[suhu, asap, gas]])
                    scaled_features = scaler.transform(features)
                    
                    input_for_lstm = scaled_features.reshape(1, 1, 3)  # samples, time steps, features
                    
                    prediction = model.predict(input_for_lstm)
                    predicted_prob = prediction[0]
                    
                    # Determine risk category from prediction
                    risk_classes = ['Aman', 'Siaga', 'Bahaya']
                    predicted_index = np.argmax(predicted_prob)
                    hasil = risk_classes[predicted_index]
                    
                except Exception as model_err:
                    print(f"Error loading model or predicting: {model_err}")
                    hasil = self.fallback_rule_based(suhu, asap, gas)
            else:
                hasil = self.fallback_rule_based(suhu, asap, gas)

            self.result_label.config(text=f"Hasil Prediksi: {hasil}")

        except Exception as e:
            messagebox.showerror("Input Error", f"Terjadi kesalahan input:\n{e}")

    def fallback_rule_based(self, suhu, asap, gas):
        # Simple fallback rule logic:
        if suhu > 30 or asap > 100 or gas > 50:
            return "Bahaya"
        elif suhu > 26 or asap > 50 or gas > 10:
            return "Siaga"
        else:
            return "Aman"

    def toggle_custom_entry(self):
        if self.periode_var.get() == '1 Periode':
            self.custom_entry.grid()
        else:
            self.custom_entry.grid_remove()

    def update_data(self):
        df = self.master.shared_data
        if df is None:
            return
        self.update_chart()

    def update_chart(self):
        df = self.master.shared_data
        if df is None or df.empty:
            messagebox.showwarning("Peringatan", "Tidak ada data yang tersedia untuk ditampilkan")
            return

        self.ax.clear()
        try:
            if 'Waktu' not in df.columns or 'Status' not in df.columns:
                raise ValueError("Kolom 'Waktu' atau 'Status' tidak ditemukan dalam data")

            data = df.copy()
            data['Waktu'] = pd.to_datetime(data['Waktu'], errors='coerce')
            data.dropna(subset=['Waktu'], inplace=True)

            # Group by day for daily predictions
            data['Tanggal'] = data['Waktu'].dt.date
            
            # Count by status
            status_counts = pd.crosstab(data['Tanggal'], data['Status'])
            
            # Ensure all status columns exist
            for status in ['Aman', 'Siaga', 'Bahaya']:
                if status not in status_counts.columns:
                    status_counts[status] = 0
            
            dates = status_counts.index.astype(str).tolist()
            bottom = np.zeros(len(dates))
            colors = {'Aman': 'green', 'Siaga': 'orange', 'Bahaya': 'red'}
            
            for status in ['Aman', 'Siaga', 'Bahaya']:
                values = status_counts[status].values
                self.ax.bar(dates, values, label=status, bottom=bottom, color=colors[status])
                bottom += values
            
            self.ax.set_title("Jumlah Status Kebakaran per Hari", fontsize=11, color=TEXT_COLOR)
            self.ax.set_xlabel("Tanggal", fontsize=10, color=TEXT_COLOR)
            self.ax.set_ylabel("Jumlah Kasus", fontsize=10, color=TEXT_COLOR)
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.5)
            self.ax.tick_params(colors=TEXT_COLOR)
            self.ax.tick_params(axis='x', rotation=45)
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Gagal menampilkan grafik:\n{e}")


class EvaluasiFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)

        tk.Label(self, text="Evaluasi Model Prediksi", 
                font=('Times New Roman', 16, 'bold'), bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=10)

        frame = tk.Frame(self, bg=BG_COLOR)
        frame.pack(pady=5)

        tk.Label(frame, text="Metode Evaluasi:", font=('Times New Roman', 12), 
                bg=BG_COLOR, fg=TEXT_COLOR).grid(row=0, column=0, sticky='w')
        
        self.method_var = tk.StringVar(value="LSTM")
        self.method_combo = ttk.Combobox(frame, textvariable=self.method_var,
                                       values=["LSTM"],
                                       font=('Times New Roman', 12), state="readonly", width=15)
        self.method_combo.grid(row=0, column=1, sticky='w')

        self.eval_btn = tk.Button(frame, text="Evaluasi", font=('Times New Roman', 12), 
                                command=self.load_evaluation_data,
                                bg=ELEMENT_COLOR, fg=TEXT_COLOR, activebackground=ELEMENT_COLOR, activeforeground=TEXT_COLOR)
        self.eval_btn.grid(row=1, columnspan=2, pady=10)

        self.visualization_frame = tk.Frame(self, bg=BG_COLOR)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.cm_frame = tk.Frame(self.visualization_frame, bg=BG_COLOR)
        self.cm_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.hist_frame = tk.Frame(self.visualization_frame, bg=BG_COLOR)
        self.hist_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.cm_figure = plt.Figure(figsize=(4, 3), dpi=100, facecolor=BG_COLOR)
        self.cm_ax = self.cm_figure.add_subplot(111)
        self.cm_ax.set_facecolor(BG_COLOR)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_figure, self.cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.hist_figure = plt.Figure(figsize=(4, 3), dpi=100, facecolor=BG_COLOR)
        self.hist_ax = self.hist_figure.add_subplot(111)
        self.hist_ax.set_facecolor(BG_COLOR)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, self.hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.analysis_label = tk.Label(self, text="Hasil Analisis", 
                                     font=('Times New Roman', 14, 'bold'), 
                                     bg=BG_COLOR, fg=TEXT_COLOR)
        self.analysis_label.pack(pady=(10, 0))

        self.result_label = tk.Text(self, font=('Times New Roman', 13), 
                                  bg=ELEMENT_COLOR, fg=TEXT_COLOR, wrap='word', height=6)
        self.result_label.pack(padx=10, pady=(5, 10), fill=tk.X)
        self.result_label.tag_configure("bold", font=('Times New Roman', 13, 'bold'))
        self.result_label.config(state=tk.DISABLED)

    def update_data(self):
        # Not needed now, but stub for interface consistency
        pass

    def load_evaluation_data(self):
        df = self.master.shared_data
        if df is None or df.empty:
            messagebox.showwarning("Peringatan", "Tidak ada data yang tersedia untuk evaluasi")
            return

        try:
            features = ['Suhu (°C)', 'Asap (ppm)', 'Gas (ppm)']
            # Create features and labels if missing for simulation
            for col in features:
                if col not in df.columns:
                    df[col] = np.random.normal(25, 5, len(df))

            if 'Status' not in df.columns:
                conditions = [
                    (df['Suhu (°C)'] > 30) | (df['Asap (ppm)'] > 100) | (df['Gas (ppm)'] > 50),
                    (df['Suhu (°C)'] > 26) | (df['Gas (ppm)'] > 10),
                    True
                ]
                choices = ['Bahaya', 'Siaga', 'Aman']
                df['Status'] = np.select(conditions, choices, default='Aman')  # Added default parameter

            # Convert Status to numeric labels first to avoid dtype issues
            le = LabelEncoder()
            y = le.fit_transform(df['Status'])
            X = df[features].values  # Make sure X is a numpy array

            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

            model_path = 'fire_detection_lstm.h5'
            scaler_path = 'feature_scaler.pkl'

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = load_model(model_path)

                # reshape X_test for LSTM input
                X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

                y_pred_prob = model.predict(X_test_reshaped)
                y_pred = np.argmax(y_pred_prob, axis=1)

                # Make sure y_pred and y_test have same data type
                y_pred = y_pred.astype(int)
                y_test = y_test.astype(int)
                
                cm = confusion_matrix(y_test, y_pred)
                
                self.cm_ax.clear()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
                disp.plot(ax=self.cm_ax, cmap='Blues', colorbar=False)
                self.cm_ax.set_title("Confusion Matrix: LSTM", fontsize=10, color=TEXT_COLOR)
                self.cm_ax.tick_params(colors=TEXT_COLOR)

                self.hist_ax.clear()
                for i, class_name in enumerate(le.classes_):
                    mask = y_test == i
                    if np.any(mask):
                        self.hist_ax.hist(y_pred_prob[mask, i], bins=10, alpha=0.7, label=f"True {class_name}")
                self.hist_ax.set_title("Prediction Probability Distribution", fontsize=10, color=TEXT_COLOR)
                self.hist_ax.set_xlabel("Probability", fontsize=8, color=TEXT_COLOR)
                self.hist_ax.set_ylabel("Frequency", fontsize=8, color=TEXT_COLOR)
                self.hist_ax.legend(fontsize=8)
                self.hist_ax.tick_params(colors=TEXT_COLOR)

                accuracy = np.mean(y_pred == y_test) * 100
                cr = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

                self.cm_canvas.draw()
                self.hist_canvas.draw()

                self.result_label.config(state=tk.NORMAL)
                self.result_label.delete('1.0', tk.END)
                self.result_label.insert(tk.END, "Hasil evaluasi untuk model ")
                self.result_label.insert(tk.END, "LSTM", "bold")
                self.result_label.insert(tk.END, f" menunjukkan akurasi sebesar {accuracy:.2f}%.\n\n")

                for cls_name in le.classes_:
                    precision = cr[cls_name]['precision'] * 100
                    recall = cr[cls_name]['recall'] * 100
                    self.result_label.insert(tk.END, f"{cls_name}: Precision {precision:.1f}%, Recall {recall:.1f}%\n")

                self.result_label.config(state=tk.DISABLED)
            else:
                messagebox.showinfo("Info", "Model LSTM belum tersedia. Silakan lakukan training terlebih dahulu.")

        except Exception as e:
            messagebox.showerror("Error", f"Gagal melakukan evaluasi:\n{e}")
            print(e)  # Print the full error for debugging


class TrainingFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=BG_COLOR)

        tk.Label(self, text="Training Model LSTM", 
                font=('Times New Roman', 16, 'bold'), 
                bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=10)

        data_frame = tk.Frame(self, bg=BG_COLOR)
        data_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(data_frame, text="Upload Dataset Excel", 
                 font=('Times New Roman', 12), 
                 command=self.upload_file,
                 bg=ELEMENT_COLOR, fg=TEXT_COLOR,
                 activebackground=ELEMENT_COLOR, activeforeground=TEXT_COLOR).pack(side=tk.LEFT, padx=5)
                 
        self.file_label = tk.Label(data_frame, text="Belum ada file yang dipilih", 
                                 bg=BG_COLOR, fg=TEXT_COLOR,
                                 font=('Times New Roman', 12))
        self.file_label.pack(side=tk.LEFT, padx=5)

        tk.Button(self, text="Mulai Training", 
                 font=('Times New Roman', 12), 
                 command=self.start_training,
                 bg=ELEMENT_COLOR, fg=TEXT_COLOR,
                 activebackground=ELEMENT_COLOR, activeforeground=TEXT_COLOR).pack(pady=10)

        self.log_text = tk.Text(self, height=12, font=('Times New Roman', 11), 
                               bg=ELEMENT_COLOR, fg=TEXT_COLOR)
        self.log_text.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)

        self.df = None
        self.model = None
        self.scaler = None
        self.le = None

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
                df.columns = df.columns.str.strip()

                required_cols = ['Waktu', 'Suhu (°C)', 'Asap (ppm)', 'Gas (ppm)']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    messagebox.showerror("Error", f"Kolom berikut tidak ditemukan: {', '.join(missing_cols)}")
                    return

                df['Waktu'] = pd.to_datetime(df['Waktu'], errors='coerce')
                if df['Waktu'].isnull().any():
                    messagebox.showerror("Error", "Beberapa nilai pada kolom 'Waktu' tidak valid (bukan waktu).")
                    return

                if df.empty:
                    messagebox.showwarning("Peringatan", "File berhasil dibaca tetapi tidak mengandung data.")
                    return

                self.df = df
                self.file_label.config(text=file_path.split("/")[-1])
                self.log_text.insert(tk.END, f"File '{file_path.split('/')[-1]}' berhasil diunggah.\n")
                self.master.shared_data = df

            except Exception as e:
                messagebox.showerror("Error", f"Gagal membuka file Excel:\n{e}")

    def start_training(self):
        if self.df is None:
            messagebox.showwarning("Peringatan", "Silakan unggah data terlebih dahulu.")
            return

        self.log_text.insert(tk.END, "Mulai proses training model LSTM...\n")

        try:
            features = ['Suhu (°C)', 'Asap (ppm)', 'Gas (ppm)']

            # Risk classification based on rules
            def classify_risk(row):
                if row['Suhu (°C)'] > 30 or row['Asap (ppm)'] > 100 or row['Gas (ppm)'] > 50:
                    return 'Bahaya'
                elif row['Suhu (°C)'] > 26 or row['Asap (ppm)'] > 50 or row['Gas (ppm)'] > 10:
                    return 'Siaga'
                else:
                    return 'Aman'

            df = self.df.copy()
            df['Status'] = df.apply(classify_risk, axis=1)

            X = df[features].values
            y = df['Status'].values

            self.le = LabelEncoder()
            y_encoded = self.le.fit_transform(y)
            y_categorical = tf.keras.utils.to_categorical(y_encoded)

            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)

            X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])  # For LSTM [samples, time steps, features]

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.3, random_state=42)

            model = Sequential()
            model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(Dense(y_categorical.shape[1], activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            self.log_text.insert(tk.END, f"Training selesai dengan akurasi: {accuracy * 100:.2f}%\n")

            # Save model and scaler
            model.save('fire_detection_lstm.h5')
            joblib.dump(self.scaler, 'feature_scaler.pkl')
            joblib.dump(self.le, 'label_encoder.pkl')

            self.model = model

        except Exception as e:
            messagebox.showerror("Error", f"Gagal saat proses training:\n{e}")

class CropFailureApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crop Fire Risk Prediction Dashboard")
        self.geometry("1100x700")
        self.configure(bg=BG_COLOR)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=BG_COLOR)
        style.configure('TNotebook.Tab', background=ELEMENT_COLOR, foreground=TEXT_COLOR)
        style.map('TNotebook.Tab', background=[('selected', ELEMENT_COLOR)])

        self.shared_data = None

        notebook = ttk.Notebook(self)

        self.dashboard = DashboardFrame(notebook)
        self.evaluasi = EvaluasiFrame(notebook)
        self.training = TrainingFrame(notebook)

        notebook.add(self.dashboard, text="Dashboard & Input Manual")
        notebook.add(self.evaluasi, text="Evaluasi")
        notebook.add(self.training, text="Training")

        notebook.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = CropFailureApp()
    app.mainloop()