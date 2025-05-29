import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from flute_data import FluteData
from flute_operations import FluteOperations

DEFAULT_DATA_JSON_DIR = "../data_json"

class TraditionalTextEditor(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Traditional Editor")
        self.geometry("800x600")
        self.filename = None
        self.create_widgets()

    def create_widgets(self):
        # Toolbar with traditional buttons
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        btn_open = ttk.Button(toolbar, text="Open", command=self.open_file)
        btn_open.pack(side=tk.LEFT, padx=2, pady=2)
        btn_save = ttk.Button(toolbar, text="Save", command=self.save_file)
        btn_save.pack(side=tk.LEFT, padx=2, pady=2)
        btn_save_as = ttk.Button(toolbar, text="Save As", command=self.save_as)
        btn_save_as.pack(side=tk.LEFT, padx=2, pady=2)
        btn_close = ttk.Button(toolbar, text="Close", command=self.close_file)
        btn_close.pack(side=tk.LEFT, padx=2, pady=2)
        btn_exit = ttk.Button(toolbar, text="Exit", command=self.exit_editor)
        btn_exit.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Text widget with scrollbars for editing content
        self.text = tk.Text(self, wrap=tk.NONE)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.text.configure(xscrollcommand=hsb.set)

    def open_file(self):
        file_path = filedialog.askopenfilename(title="Open File", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            self.filename = file_path
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text.delete("1.0", tk.END)
                self.text.insert(tk.END, content)
            except Exception as e:
                messagebox.showerror("Error", f"Error opening file:\n{e}")

    def save_file(self):
        if not self.filename:
            self.save_as()
        else:
            try:
                with open(self.filename, "w", encoding="utf-8") as f:
                    f.write(self.text.get("1.0", tk.END))
                messagebox.showinfo("Saved", "File saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving file:\n{e}")

    def save_as(self):
        file_path = filedialog.asksaveasfilename(title="Save As", defaultextension=".json",
                                                 filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if file_path:
            self.filename = file_path
            self.save_file()

    def close_file(self):
        self.filename = None
        self.text.delete("1.0", tk.END)

    def exit_editor(self):
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flute Analysis")
        self.geometry("1200x800")
        
        # Top bar to place the "Exit" button at the top right
        top_bar = ttk.Frame(self)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        exit_button = ttk.Button(top_bar, text="Exit", command=self.close_app)
        exit_button.pack(side=tk.RIGHT)
        
        # Frame for selecting flute(s) and opening the popup editor
        selection_frame = ttk.Frame(self)
        selection_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(selection_frame, text="Select one or more flutes:").pack(side=tk.LEFT)
        
        self.data_dir = DEFAULT_DATA_JSON_DIR # Se puede hacer configurable
        browse_button = ttk.Button(selection_frame, text="Browse Data Dir", command=self.browse_data_directory)
        browse_button.pack(side=tk.LEFT, padx=5)

        # Listbox for multiple selection
        self.flute_list = []
        self.flute_listbox = tk.Listbox(selection_frame, selectmode=tk.EXTENDED, height=6)
        self.populate_flute_listbox() # Llenar inicialmente
        self.flute_listbox.pack(side=tk.LEFT, padx=10)
        
        # Mover la inicialización de flute_list y flute_listbox a un método
        if not self.flute_list:
            self.flute_listbox.insert(tk.END, "No flutes in default directory")
        
        load_button = ttk.Button(selection_frame, text="Load", command=self.load_flutes)
        load_button.pack(side=tk.LEFT, padx=10)
        
        # Button to open the traditional editor in a popup
        editor_button = ttk.Button(selection_frame, text="JSON Editor", command=self.open_json_editor)
        editor_button.pack(side=tk.LEFT, padx=10)
        
        # Create a Notebook with tabs for different graphs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=1, fill='both')
        
        # New tab "Parts" and others
        self.geometry_frame = ttk.Frame(self.notebook)
        self.parts_frame = ttk.Frame(self.notebook)
        self.inharmonic_frame = ttk.Frame(self.notebook)
        self.moc_frame = ttk.Frame(self.notebook)
        self.admittance_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.geometry_frame, text="Profile")
        self.notebook.add(self.parts_frame, text="Parts")
        self.notebook.add(self.admittance_frame, text="Admittance")
        self.notebook.add(self.inharmonic_frame, text="Inharmonicity")
        self.notebook.add(self.moc_frame, text="MOC")
        self.bi_espe_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.bi_espe_frame, text="B_I & ESPE")
        
        # Admittance tab: Combobox to choose the note
        note_selection_frame = ttk.Frame(self.admittance_frame)
        note_selection_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        ttk.Label(note_selection_frame, text="Select note:").pack(side=tk.LEFT)
        self.note_var = tk.StringVar()
        self.note_combobox = ttk.Combobox(note_selection_frame, textvariable=self.note_var, state="readonly")
        self.note_combobox.pack(side=tk.LEFT, padx=5)
        self.note_combobox.bind("<<ComboboxSelected>>", self.update_admittance_plot)
        
        # Frame to contain the Admittance plot
        self.admittance_plot_frame = ttk.Frame(self.admittance_frame)
        self.admittance_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Variables to store loaded flute analysis instances
        self.flute_ops_list = []  # List of FluteOperations
        self.acoustic_analysis_list = []  # List of tuples (acoustic_analysis, flute_model)
        self.finger_frequencies = {}  # Taken from the first loaded flute

    def browse_data_directory(self):
        dir_path = filedialog.askdirectory(initialdir=".", title="Select Flute Data Directory")
        if dir_path:
            self.data_dir = dir_path
            self.populate_flute_listbox()

    def populate_flute_listbox(self):
        self.flute_listbox.delete(0, tk.END)
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            self.flute_list = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            for flute in self.flute_list:
                self.flute_listbox.insert(tk.END, flute)
        else:
            self.flute_listbox.insert(tk.END, f"Directory not found: {self.data_dir}")
    def load_flutes(self):
        selected_indices = self.flute_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one flute.")
            return
        selected_flautas = [self.flute_list[i] for i in selected_indices]
        
        self.flute_ops_list = []
        self.acoustic_analysis_list = []
        self.finger_frequencies = {}
        
        for flute in selected_flautas:
            data_path = os.path.join(self.data_dir, flute)
            try:
                flute_data = FluteData(data_path)
                flute_ops = FluteOperations(flute_data)
                self.flute_ops_list.append(flute_ops)
                self.acoustic_analysis_list.append((flute_data.acoustic_analysis, flute_data.data.get("Flute Model", flute)))
                if not self.finger_frequencies:
                    self.finger_frequencies = flute_data.finger_frequencies
            except Exception as e:
                messagebox.showerror("Error", f"Could not load flute {flute}: {e}")
        
        if not self.flute_ops_list:
            return
        
        self.update_parts_plot()
        self.update_geometry_plot()
        self.update_inharmonic_plot()
        self.update_moc_plot()
        self.update_admittance_note_options()
        self.update_bi_espe_plot()

    def _setup_plot_canvas(self, parent_frame: ttk.Frame, fig: plt.Figure):
        """Helper para limpiar frame y dibujar canvas."""
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # Aplicar estilo común a los ejes de la figura
        for ax in fig.get_axes():
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.minorticks_on()
            ax.grid(which='minor', linestyle=':', linewidth=0.3, alpha=0.5)

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return canvas

    def update_parts_plot(self):
        if not self.flute_ops_list: return
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes.flatten()

        flute_names = [ops.flute_data.data.get("Flute Model", f"Flute {i}") 
                    for i, ops in enumerate(self.flute_ops_list, start=1)]

        for i, flute_ops in enumerate(self.flute_ops_list):
            flute_ops.plot_individual_parts(
                ax=ax,
                flute_names=[flute_names[i]],  # Ensures each flute has its own name in the legend
                flute_color=colors[i % len(colors)]
            )

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(", ".join(flute_names), fontsize=12)
        self._setup_plot_canvas(self.parts_frame, fig)

    def update_geometry_plot(self):
        if not self.flute_ops_list: return
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        linestyles = ['-', '--', '-.', ':']
        fig, ax = plt.subplots(figsize=(20, 12))
        for i, flute_ops in enumerate(self.flute_ops_list):
            flute_ops.plot_combined_flute_data(ax=ax, flute_color=colors[i % len(colors)], flute_style=linestyles[i % len(linestyles)])
        self._setup_plot_canvas(self.geometry_frame, fig)

    def update_inharmonic_plot(self):
        if not self.flute_ops_list or not self.acoustic_analysis_list: return
        notes = list(self.finger_frequencies.keys()) if self.finger_frequencies else []
        # Usar el método público si se renombró en FluteOperations
        fig = self.flute_ops_list[0].plot_summary_cents_differences(self.acoustic_analysis_list, notes)
        self._setup_plot_canvas(self.inharmonic_frame, fig)

    def update_moc_plot(self):
        if not self.flute_ops_list or not self.acoustic_analysis_list: return
        notes = list(self.finger_frequencies.keys()) if self.finger_frequencies else []
        fig = self.flute_ops_list[0].plot_moc_summary(self.acoustic_analysis_list, self.finger_frequencies, notes)
        self._setup_plot_canvas(self.moc_frame, fig)

    def update_admittance_note_options(self):
        if not self.flute_ops_list: # No hay flautas cargadas
            self.note_combobox['values'] = []
            self.note_var.set("")
            # Limpiar el plot de admitancia si existe
            for widget in self.admittance_plot_frame.winfo_children():
                widget.destroy()
            return
            
        if self.finger_frequencies:
            notes = list(self.finger_frequencies.keys())
            self.note_combobox['values'] = notes
            self.note_var.set(notes[0])
            self.update_admittance_plot(None)

    def update_admittance_plot(self, event):
        selected_note = self.note_var.get()
        if not selected_note or not self.flute_ops_list:
            return
        # Usar el método público si se renombró en FluteOperations
        fig = self.flute_ops_list[0].plot_individual_admittance_analysis(self.acoustic_analysis_list, selected_note)
        self._setup_plot_canvas(self.admittance_plot_frame, fig)
    
    def open_json_editor(self):
        editor = TraditionalTextEditor(self)
        editor.grab_set()

    def close_app(self):
        plt.close('all')
        self.destroy()

    def update_bi_espe_plot(self):
        if not self.flute_ops_list or not self.acoustic_analysis_list: return
        notes = list(self.finger_frequencies.keys()) if self.finger_frequencies else []
        fig = self.flute_ops_list[0].plot_bi_espe_summary(self.acoustic_analysis_list, self.finger_frequencies, notes)
        self._setup_plot_canvas(self.bi_espe_frame, fig)

if __name__ == "__main__":
    app = App()
    app.mainloop()