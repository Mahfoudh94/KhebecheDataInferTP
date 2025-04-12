import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from customtkinter import CTk, CTkLabel, CTkButton, filedialog, CTkFrame, \
    CTkOptionMenu, CTkEntry, StringVar, CTkScrollableFrame, CTkCheckBox, BooleanVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import LabelEncoder

from models import ModelFactory
from readers import ReaderFactory


class MLApp(CTk):
    def __init__(self):
        super().__init__()
        self.title("ML Model Selector")
        self.geometry("800x600")

        self.model_type = StringVar(value=None)
        self.target_column = StringVar()
        self.input_data = None
        self.model = None
        self.encoder_type = StringVar(value="OneHot")
        self.encoding_vars = {}
        self.exclude_vars = {}

        self.setup_ui()

    def setup_ui(self):
        left_frame = CTkFrame(self)
        left_frame.pack(side="left", fill="y", padx=10, pady=10)

        right_frame = CTkFrame(self)
        right_frame.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        self.load_button = CTkButton(left_frame, text="Load data", command=self.load_data)
        self.load_button.pack(pady=10)

        self.model_menu = CTkOptionMenu(left_frame, values=list(ModelFactory.get_keys()), variable=self.model_type)
        self.model_menu.pack(pady=10)

        self.target_menu = CTkOptionMenu(left_frame, values=[], variable=self.target_column)
        self.target_menu.pack(pady=10)

        self.train_button = CTkButton(left_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.encoding_frame = CTkScrollableFrame(left_frame, width=350, height=300)
        self.encoding_frame.pack(pady=10, fill="both", expand=True)

        self.plot_frame = CTkFrame(right_frame)
        self.plot_frame.pack(expand=True, fill="both", pady=10)

        self.test_inputs_outer_frame = CTkFrame(right_frame)
        self.test_inputs_outer_frame.pack(pady=10, fill="x")
        self.test_inputs_label = CTkLabel(self.test_inputs_outer_frame, text="Test Inputs")
        self.test_inputs_label.pack()
        self.test_inputs_frame = CTkScrollableFrame(self.test_inputs_outer_frame, width=200)
        self.test_inputs_frame.pack()
        self.predict_button = CTkButton(self.test_inputs_outer_frame, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = CTkLabel(self.test_inputs_outer_frame, text="Prediction result will appear here")
        self.result_label.pack(pady=10)

    def load_data(self):
        extensions = ReaderFactory.get_extensions()
        file_path = filedialog.askopenfilename(filetypes=extensions)

        if not file_path:
            return

        file_ext = file_path.split(".")[-1]
        file_reader = ReaderFactory.get_instance_by_extension(file_ext)

        if not file_reader:
            return

        self.input_data = file_reader.read(file_path)
        self.target_menu.configure(values=list(self.input_data.columns))
        if self.input_data.columns.size > 0:
            self.target_column.set(self.input_data.columns[-1])

        self.update_encoding_options()

    def update_encoding_options(self):
        for widget in self.encoding_frame.winfo_children():
            widget.destroy()
        for widget in self.test_inputs_frame.winfo_children():
            widget.destroy()
        self.encoding_vars.clear()
        self.exclude_vars.clear()

        for col in self.input_data.columns:
            enc_var = BooleanVar(value=False)
            ex_var = BooleanVar(value=False)
            col_type = str(self.input_data[col].dtype)
            label = f"{col} - {col_type}"

            frame = CTkFrame(self.encoding_frame)
            frame.pack(fill="x", padx=2, pady=2)

            chk_enc = CTkCheckBox(frame, text=label, variable=enc_var)
            chk_enc.grid(column=0)

            chk_ex = CTkCheckBox(frame, text="Exclude", variable=ex_var)
            chk_ex.grid(column=1)

            self.encoding_vars[col] = enc_var
            self.exclude_vars[col] = ex_var

    def apply_encoding(self, X):
        selected_cols = [col for col, var in self.encoding_vars.items() if var.get() and col in X.columns]
        encoder = self.encoder_type.get()
        X_encoded = X.copy()

        if encoder == "OneHot":
            X_encoded = pd.get_dummies(X_encoded, columns=selected_cols)
        elif encoder == "Label":
            for col in selected_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col])

        # Handle leftover object columns automatically
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = pd.factorize(X_encoded[col])[0]

        print(X_encoded.dtypes)

        return X_encoded

    def update_test_inputs(self):
        for widget in self.test_inputs_frame.winfo_children():
            widget.destroy()
        self.test_entries = {}
        for col in self.feature_order:
            entry = CTkEntry(self.test_inputs_frame, placeholder_text=f"{col}")
            entry.pack(padx=2, pady=2, fill="x")
            self.test_entries[col] = entry

    def train_model(self):
        if self.input_data is None:
            return
        target = self.target_column.get()
        if target not in self.input_data.columns:
            return

        X = self.input_data.drop(columns=[target])
        excluded = [col for col, var in self.exclude_vars.items() if var.get() and col in X.columns]
        X = X.drop(columns=excluded)
        self.feature_order = X.columns.tolist()
        X = self.apply_encoding(X)
        y = self.input_data[target].values

        self.model = ModelFactory.create(self.model_type.get())
        if self.model.is_classifier:
            self.y_encoder = LabelEncoder()
            y = self.y_encoder.fit_transform(y)

        self.model.train(X, y)

        self.update_test_inputs()
        self.plot_history()

    def plot_history(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        history = self.model.history
        if hasattr(history, "history"):
            history = history.history  # Keras-style

        ax[0].plot(history.get('loss', []), label='Loss')
        ax[0].plot(history.get('val_loss', []), label='Val Loss')
        ax[0].legend()
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        if 'accuracy' in history:
            ax[1].plot(history.get('accuracy', []), label='Accuracy')
            ax[1].plot(history.get('val_accuracy', []), label='Val Accuracy')
            ax[1].set_ylabel("Accuracy")
        elif 'mae' in history:
            ax[1].plot(history.get('mae', []), label='MAE')
            ax[1].plot(history.get('val_mae', []), label='Val MAE')
            ax[1].set_ylabel("MAE")

        ax[1].legend()
        ax[1].set_title("Metrics")
        ax[1].set_xlabel("Epoch")

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

    def predict(self):
        if not self.model:
            return
        try:
            values = []
            for col in self.feature_order:
                entry = self.test_entries.get(col)
                if entry:
                    values.append(float(entry.get()))
            input_values = np.array(values).reshape(1, -1)
            prediction = self.model.predict(input_values)
            result = prediction
            if isinstance(prediction, np.ndarray):
                # Flatten nested predictions like [[value]] → value
                if prediction.ndim > 1 and prediction.shape[0] == 1:
                    result = prediction[0]
                if isinstance(result, np.ndarray) and result.shape[0] == 1:
                    result = result[0]
            self.result_label.configure(text=f"Prediction: {result}")
        except Exception as e:
            self.result_label.configure(text=f"Error: {e}")


root = MLApp()
root.mainloop()
