import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pathlib import Path 
from typing import Optional, List, Tuple, Dict 

from flute_data import FluteData
from flute_operations import FluteOperations
from constants import BASE_COLORS, LINESTYLES, FLUTE_PARTS_ORDER

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_JSON_DIR = SCRIPT_DIR / "data_json" 
if not DEFAULT_DATA_JSON_DIR.exists():
     DEFAULT_DATA_JSON_DIR = SCRIPT_DIR.parent / "data_json"

class TraditionalTextEditor(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Editor JSON Tradicional")
        self.geometry("800x600")
        self.filename = None
        self.create_widgets()

    def create_widgets(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        btn_open = ttk.Button(toolbar, text="Abrir", command=self.open_file)
        btn_open.pack(side=tk.LEFT, padx=2, pady=2)
        btn_save = ttk.Button(toolbar, text="Guardar", command=self.save_file)
        btn_save.pack(side=tk.LEFT, padx=2, pady=2)
        btn_save_as = ttk.Button(toolbar, text="Guardar Como", command=self.save_as)
        btn_save_as.pack(side=tk.LEFT, padx=2, pady=2)
        btn_close = ttk.Button(toolbar, text="Cerrar Archivo", command=self.close_file)
        btn_close.pack(side=tk.LEFT, padx=2, pady=2)
        btn_exit = ttk.Button(toolbar, text="Salir Editor", command=self.exit_editor)
        btn_exit.pack(side=tk.LEFT, padx=2, pady=2)
        
        self.text = tk.Text(self, wrap=tk.WORD)
        self.text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.configure(yscrollcommand=vsb.set)
        
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.text.configure(xscrollcommand=hsb.set)

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="Abrir Archivo JSON", 
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            initialdir=str(DEFAULT_DATA_JSON_DIR) 
        )
        if file_path:
            self.filename = file_path
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text.delete("1.0", tk.END)
                self.text.insert(tk.END, content)
                self.title(f"Editor JSON - {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error Abriendo Archivo", f"No se pudo abrir el archivo:\n{e}")

    def save_file(self):
        if not self.filename:
            self.save_as()
        else:
            try:
                with open(self.filename, "w", encoding="utf-8") as f:
                    f.write(self.text.get("1.0", tk.END).strip() + "\n")
                messagebox.showinfo("Guardado", f"Archivo guardado exitosamente:\n{self.filename}")
            except Exception as e:
                messagebox.showerror("Error Guardando Archivo", f"No se pudo guardar el archivo:\n{e}")

    def save_as(self):
        initial_dir_path = os.path.dirname(self.filename) if self.filename else str(DEFAULT_DATA_JSON_DIR)
        file_path = filedialog.asksaveasfilename(
            title="Guardar Archivo Como", 
            defaultextension=".json",
            filetypes=[("Archivos JSON", "*.json"), ("Todos los archivos", "*.*")],
            initialdir=initial_dir_path
        )
        if file_path:
            self.filename = file_path
            self.save_file()
            self.title(f"Editor JSON - {os.path.basename(file_path)}")

    def close_file(self):
        self.filename = None
        self.text.delete("1.0", tk.END)
        self.title("Editor JSON Tradicional")

    def exit_editor(self):
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis de Flautas")
        self.geometry("1200x800")
        
        self.data_dir = str(DEFAULT_DATA_JSON_DIR)
        self.flute_list_paths: List[str] = [] 
        self.flute_checkbox_vars: Dict[str, tk.BooleanVar] = {} # For Checkbuttons

        print(f"DEBUG: App.__init__ - ID(self): {id(self)}")
        print(f"DEBUG: App.__init__ - BEFORE populate - self.flute_list_paths: {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")
        print(f"DEBUG: App.__init__ - BEFORE populate - self.flute_checkbox_vars: {self.flute_checkbox_vars}")


        style = ttk.Style(self)
        # style.theme_use('clam') 

        top_bar = ttk.Frame(self)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        exit_button = ttk.Button(top_bar, text="Salir de Aplicación", command=self.close_app)
        exit_button.pack(side=tk.RIGHT)
        
        selection_outer_frame = ttk.LabelFrame(self, text="Selección de Flautas", padding=(10,5))
        # No llenar en X, dejar que tome el ancho de su contenido. pady reducido.
        selection_outer_frame.pack(side=tk.TOP, fill=tk.NONE, expand=False, padx=10, pady=(5,2))
        
        # Directory selection row
        dir_selection_row = ttk.Frame(selection_outer_frame)
        dir_selection_row.pack(fill=tk.X, pady=(0,5))
        ttk.Label(dir_selection_row, text="Directorio de Datos:").pack(side=tk.LEFT, padx=(0,5))
        self.data_dir_label = ttk.Label(dir_selection_row, text=self.data_dir, relief="sunken", width=50)
        self.data_dir_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        browse_button = ttk.Button(dir_selection_row, text="Cambiar Directorio", command=self.browse_data_directory)
        browse_button.pack(side=tk.LEFT)

        # Flute selection area (with scrollable checkbuttons)
        flute_list_area_frame = ttk.Frame(selection_outer_frame)
        # No expandir, dejar que tome la altura necesaria o una fija
        flute_list_area_frame.pack(fill=tk.X, expand=False, pady=(2,0)) 

        ttk.Label(flute_list_area_frame, text="Flautas Disponibles:").pack(anchor="w", padx=(0,5))

        # Canvas for scrollable checkbuttons - limitar su altura
        # Establecer una altura fija más pequeña, por ejemplo, para 2-3 items.
        # La altura de un checkbutton es aprox 20-25px.
        self.flute_selection_canvas = tk.Canvas(flute_list_area_frame, borderwidth=0, background="#ffffff", height=60) # Altura para aprox. 3 items
        self.flute_selection_frame = ttk.Frame(self.flute_selection_canvas) # Frame to hold checkbuttons
        
        self.flute_scrollbar = ttk.Scrollbar(flute_list_area_frame, orient="vertical", command=self.flute_selection_canvas.yview)
        self.flute_selection_canvas.configure(yscrollcommand=self.flute_scrollbar.set)

        self.flute_scrollbar.pack(side="right", fill="y")
        # El canvas se expandirá horizontalmente, pero su altura está fijada por el parámetro 'height'
        self.flute_selection_canvas.pack(side="left", fill=tk.X, expand=True)
        self.canvas_frame_id = self.flute_selection_canvas.create_window((0, 0), window=self.flute_selection_frame, anchor="nw")

        self.flute_selection_frame.bind("<Configure>", self._on_frame_configure)
        self.flute_selection_canvas.bind("<Configure>", self._on_canvas_configure)


        # Buttons frame (Load, Editor)
        button_frame = ttk.Frame(selection_outer_frame) # pady reducido
        button_frame.pack(fill=tk.X, pady=(5,0))
        load_button = ttk.Button(button_frame, text="Cargar Seleccionadas", command=self.load_flutes)
        load_button.pack(side=tk.LEFT, padx=(0,10))
        editor_button = ttk.Button(button_frame, text="Editor JSON", command=self.open_json_editor)
        editor_button.pack(side=tk.LEFT)
        
        self.populate_flute_selection_ui() 
        print(f"DEBUG: App.__init__ - AFTER populate - self.flute_list_paths: {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")
        print(f"DEBUG: App.__init__ - AFTER populate - self.flute_checkbox_vars: {self.flute_checkbox_vars}")
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=(2,10)) # Reducir pady top aún más
        
        self.profile_frame = ttk.Frame(self.notebook)
        self.parts_frame = ttk.Frame(self.notebook)
        self.admittance_frame = ttk.Frame(self.notebook)
        self.inharmonic_frame = ttk.Frame(self.notebook)
        self.moc_frame = ttk.Frame(self.notebook)
        self.bi_espe_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.profile_frame, text="Perfil Combinado")
        self.notebook.add(self.parts_frame, text="Partes Individuales")
        self.notebook.add(self.admittance_frame, text="Admitancia (por Nota)")
        self.notebook.add(self.inharmonic_frame, text="Inharmonicidad (Resumen)")
        self.notebook.add(self.moc_frame, text="MOC (Resumen)")
        self.notebook.add(self.bi_espe_frame, text="B_I & ESPE (Resumen)")
        
        note_selection_frame = ttk.Frame(self.admittance_frame)
        note_selection_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Label(note_selection_frame, text="Seleccionar nota:").pack(side=tk.LEFT, padx=(0,5))
        self.note_var = tk.StringVar()
        self.note_combobox = ttk.Combobox(note_selection_frame, textvariable=self.note_var, state="readonly", width=10)
        self.note_combobox.pack(side=tk.LEFT)
        self.note_combobox.bind("<<ComboboxSelected>>", self.update_admittance_plot)
        
        self.admittance_plot_frame = ttk.Frame(self.admittance_frame)
        self.admittance_plot_frame.pack(fill=tk.BOTH, expand=True)
        
        self.flute_ops_list: List[FluteOperations] = []
        self.acoustic_analysis_list_for_summary: List[Tuple[dict, str]] = []
        self.finger_frequencies_map_for_summary: Dict[str, Dict[str, float]] = {}
        self.combined_measurements_list_for_summary: List[Tuple[List[Dict[str, float]], str]] = []
        self.ordered_notes_for_summary: List[str] = []

    def _on_frame_configure(self, event=None):
        # Update scrollregion to encompass the inner frame
        self.flute_selection_canvas.configure(scrollregion=self.flute_selection_canvas.bbox("all"))

    def _on_canvas_configure(self, event=None):
        # Resize the inner frame to match the canvas width
        canvas_width = event.width # Usar el ancho del evento
        self.flute_selection_canvas.itemconfig(self.canvas_frame_id, width=canvas_width)
        # La altura del canvas ya está fijada en su creación.
        # El scrollregion se actualiza en _on_frame_configure cuando el contenido del frame interno cambia.


    def browse_data_directory(self):
        dir_path = filedialog.askdirectory(initialdir=self.data_dir, title="Seleccionar Directorio de Datos de Flautas")
        if dir_path:
            self.data_dir = dir_path
            self.data_dir_label.config(text=self.data_dir)
            self.populate_flute_selection_ui()

    def populate_flute_selection_ui(self):
        # Clear previous checkbuttons from the frame
        for widget in self.flute_selection_frame.winfo_children():
            widget.destroy()
        
        self.flute_checkbox_vars.clear()
        temp_flute_paths = [] 
        current_data_dir = Path(self.data_dir)
        
        print(f"DEBUG: populate_flute_selection_ui - ID(self): {id(self)}")
        print(f"DEBUG: populate_flute_selection_ui - Attempting to populate from: {current_data_dir}")

        if not (current_data_dir.exists() and current_data_dir.is_dir()):
            ttk.Label(self.flute_selection_frame, text=f" (Directorio no encontrado: {self.data_dir})", foreground="red").pack(anchor="w")
            self.flute_list_paths = [] 
            print(f"DEBUG: populate_flute_selection_ui - data_dir '{self.data_dir}' is invalid. flute_list_paths set to []. ID: {id(self.flute_list_paths)}")
            self._on_frame_configure() # Update scrollregion
            return

        try:
            sub_dir_names = []
            for item_name in os.listdir(current_data_dir):
                if (current_data_dir / item_name).is_dir():
                    sub_dir_names.append(item_name)
            
            print(f"DEBUG: populate_flute_selection_ui - Found subdirectories: {sub_dir_names}")

            if not sub_dir_names:
                ttk.Label(self.flute_selection_frame, text=" (No hay subdirectorios de flautas aquí)", foreground="grey").pack(anchor="w")
                self.flute_list_paths = [] 
            else:
                # Sort and store paths
                self.flute_list_paths = sorted(sub_dir_names)
                for flute_dir_name in self.flute_list_paths:
                    var = tk.BooleanVar(value=False)
                    cb = ttk.Checkbutton(self.flute_selection_frame, text=flute_dir_name, variable=var)
                    cb.pack(anchor="w", padx=5, pady=1) # pady reduced for compactness
                    self.flute_checkbox_vars[flute_dir_name] = var
            
        except OSError as e:
            ttk.Label(self.flute_selection_frame, text=f" (Error listando directorio: {e.strerror})", foreground="red").pack(anchor="w")
            self.flute_list_paths = [] 
            print(f"DEBUG: populate_flute_selection_ui - OSError: {e}. flute_list_paths set to []. ID: {id(self.flute_list_paths)}")
            self._on_frame_configure() # Update scrollregion
            return 

        self._on_frame_configure() # Update scrollregion after adding widgets
        print(f"DEBUG: populate_flute_selection_ui - FINAL self.flute_list_paths: {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")
        print(f"DEBUG: populate_flute_selection_ui - FINAL self.flute_checkbox_vars: {self.flute_checkbox_vars}")
            
    def load_flutes(self):
        print(f"DEBUG: En load_flutes - ID(self): {id(self)}")
        print(f"DEBUG: En load_flutes - self.flute_list_paths (al inicio): {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")
        print(f"DEBUG: En load_flutes - self.flute_checkbox_vars (al inicio): {self.flute_checkbox_vars}")
        
        selected_flute_dirs = []
        for flute_dir_name, var in self.flute_checkbox_vars.items():
            if var.get(): 
                selected_flute_dirs.append(flute_dir_name)
        # La condición anterior 'if flute_dir_name in self.flute_list_paths:' se eliminó

        print(f"DEBUG: En load_flutes - selected_flute_dirs (de checkboxes): {selected_flute_dirs}")

        # Reiniciar listas antes de cargar nuevas flautas
        self.flute_ops_list = []
        if not selected_flute_dirs:
            messagebox.showwarning("Sin Selección", "Por favor, seleccione al menos una flauta marcando la casilla correspondiente.")
            return
        
        self.flute_ops_list = []
        self.acoustic_analysis_list_for_summary = []
        self.finger_frequencies_map_for_summary = {}
        self.combined_measurements_list_for_summary = []
        self.ordered_notes_for_summary = []

        successful_loads = 0
        for flute_dir_name in selected_flute_dirs:
            data_path = Path(self.data_dir) / flute_dir_name
            try:
                print(f"DEBUG: load_flutes - Intentando cargar FluteData desde: {data_path}")
                flute_data_obj = FluteData(str(data_path))
                print(f"DEBUG: load_flutes - FluteData cargada para: {flute_dir_name}")
                flute_ops_obj = FluteOperations(flute_data_obj)
                
                self.flute_ops_list.append(flute_ops_obj)
                # Usar flute_model de FluteData que ya está normalizado
                flute_model_name = flute_data_obj.flute_model 
                self.acoustic_analysis_list_for_summary.append(
                    (flute_data_obj.acoustic_analysis, flute_model_name)
                )
                self.combined_measurements_list_for_summary.append(
                    (flute_data_obj.combined_measurements, flute_model_name)
                )
                if flute_data_obj.finger_frequencies: # Asegurarse que no es None
                    self.finger_frequencies_map_for_summary[flute_model_name] = flute_data_obj.finger_frequencies
                successful_loads += 1
            except Exception as e:
                detailed_error_msg = f"No se pudo cargar la flauta desde '{flute_dir_name}'.\n"
                detailed_error_msg += f"Ruta completa: {data_path}\n"
                detailed_error_msg += f"Error: {type(e).__name__}: {e}\n\n"
                detailed_error_msg += "Verifique que el directorio contenga los archivos JSON necesarios (headjoint.json, etc.) y que tengan el formato correcto."
                messagebox.showerror("Error de Carga", detailed_error_msg)
                print(f"ERROR: load_flutes - Excepción al cargar {flute_dir_name}: {e}") 
        
        if not successful_loads and selected_flute_dirs:
             messagebox.showinfo("Carga Fallida", "No se pudo cargar ninguna de las flautas seleccionadas que parecían válidas.")
             return
        
        if successful_loads > 0:
            # messagebox.showinfo("Carga Exitosa", f"Se cargaron {successful_loads} flauta(s) exitosamente.")
            print(f"INFO: Se cargaron {successful_loads} flauta(s) exitosamente.") # Log en consola en su lugar
            
            if self.finger_frequencies_map_for_summary:
                all_present_notes = set()
                for ff_map in self.finger_frequencies_map_for_summary.values():
                    all_present_notes.update(ff_map.keys())
                
                canonical_order = ["D", "D#", "E", "F", "Fs", "G", "G#", "A", "A#", "B", "C", "Cs"] 
                self.ordered_notes_for_summary = [n for n in canonical_order if n in all_present_notes]
                self.ordered_notes_for_summary.extend(sorted(list(all_present_notes - set(self.ordered_notes_for_summary))))

            self.update_all_plots()
            self.update_admittance_note_options()

    def _setup_plot_canvas(self, parent_frame: ttk.Frame, fig: plt.Figure):
        for widget in parent_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.draw()
        return canvas

    def update_all_plots(self):
        if not self.flute_ops_list: return
        self.update_profile_plot()
        self.update_parts_plot()
        self.update_inharmonic_plot()
        self.update_moc_plot()
        self.update_bi_espe_plot()

    def update_profile_plot(self):
        if not self.flute_ops_list: return
        fig, ax = plt.subplots(figsize=(12, 7))
        for i, flute_ops in enumerate(self.flute_ops_list):
            flute_model_name = flute_ops.flute_data.flute_model # Usar el modelo de FluteData
            flute_ops.plot_combined_flute_data(
                ax=ax,
                plot_label=flute_model_name,  # <--- CORRECCIÓN AQUÍ
                flute_color=BASE_COLORS[i % len(BASE_COLORS)],
                flute_style=LINESTYLES[i % len(LINESTYLES)]
            )
            # Dibujar agujeros para esta flauta
            if flute_ops.flute_data.combined_measurements:
                min_diam_profile = min(m['diameter'] for m in flute_ops.flute_data.combined_measurements if 'diameter' in m) if flute_ops.flute_data.combined_measurements else 10
                y_pos_holes = min_diam_profile - (5 + i * 0.5) # Pequeño offset para cada flauta

                current_abs_pos_holes = 0.0
                for part_name_hole in FLUTE_PARTS_ORDER:
                    part_data_hole = flute_ops.flute_data.data.get(part_name_hole, {})
                    hole_positions_rel = part_data_hole.get("Holes position", [])
                    hole_diameters_rel = part_data_hole.get("Holes diameter", [])

                    for h_pos, h_diam in zip(hole_positions_rel, hole_diameters_rel):
                        abs_hole_pos = current_abs_pos_holes + h_pos
                        # Dibujar el agujero como un marcador circular 'o'
                        marker_size_scaled = max(h_diam * 0.8, 3) # Ajustar factor de escala si es necesario
                        ax.plot(abs_hole_pos, y_pos_holes, 
                                marker='o', color=BASE_COLORS[i % len(BASE_COLORS)], 
                                markersize=marker_size_scaled, 
                                linestyle='None', alpha=0.7)

                    total_length_part = part_data_hole.get("Total length", 0.0)
                    mortise_length_part = part_data_hole.get("Mortise length", 0.0)
                    if FLUTE_PARTS_ORDER.index(part_name_hole) < len(FLUTE_PARTS_ORDER) -1 :
                        current_abs_pos_holes += (total_length_part - mortise_length_part)

        # Si hay múltiples flautas, la leyenda se generará con las etiquetas de cada plot
        if len(self.flute_ops_list) > 0 : # Solo añadir leyenda si se ploteó algo
            ax.legend(loc='best', title="Flautas")

        ax.set_title("Perfiles Combinados de Flautas")
        self._setup_plot_canvas(self.profile_frame, fig)

    def update_parts_plot(self):
        if not self.flute_ops_list: return

        fig, axes_array = plt.subplots(2, 2, figsize=(12, 10)) 
        axes_flat = list(axes_array.flatten()) # [ax_head, ax_left, ax_right, ax_foot]

        flute_names_for_title = []

        for flute_idx, flute_ops_instance in enumerate(self.flute_ops_list):
            flute_model_name = flute_ops_instance.flute_data.flute_model
            if flute_model_name not in flute_names_for_title: # Evitar duplicados en el título
                flute_names_for_title.append(flute_model_name)
            
            current_flute_color = BASE_COLORS[flute_idx % len(BASE_COLORS)]
            current_flute_style = LINESTYLES[flute_idx % len(LINESTYLES)]

            for part_idx, part_name in enumerate(FLUTE_PARTS_ORDER):
                if part_idx >= len(axes_flat): break
                
                ax_part = axes_flat[part_idx]
                adjusted_positions, diameters = flute_ops_instance._calculate_adjusted_positions(part_name, 0.0)

                if not adjusted_positions or not diameters: continue

                ax_part.plot(adjusted_positions, diameters, marker='.', linestyle=current_flute_style,
                             color=current_flute_color, markersize=3, label=flute_model_name)

                part_data_dict = flute_ops_instance.flute_data.data.get(part_name, {})
                hole_positions_part = part_data_dict.get("Holes position", [])
                hole_diameters_part = part_data_dict.get("Holes diameter", [])

                if hole_positions_part and hole_diameters_part:
                    min_diam_this_part_this_flute = min(diameters) if diameters else 0
                    y_pos_for_holes = min_diam_this_part_this_flute - (5 + flute_idx * 1.5) 

                    for h_pos, h_diam in zip(hole_positions_part, hole_diameters_part):
                        # Dibujar el agujero como un marcador circular 'o'
                        marker_size_scaled_part = max(h_diam * 0.8, 3) # Ajustar factor de escala
                        ax_part.plot(h_pos, y_pos_for_holes, 
                                     marker='o', color=current_flute_color, 
                                     markersize=marker_size_scaled_part, 
                                     linestyle='None', alpha=0.7)
                
                ax_part.set_title(f"{part_name.capitalize()}", fontsize=9)
                ax_part.set_xlabel("Posición en parte (mm)", fontsize=8)
                ax_part.set_ylabel("Diámetro (mm)", fontsize=8)
                ax_part.grid(True, linestyle=':', alpha=0.5)
                ax_part.tick_params(axis='both', which='major', labelsize=7)

        for ax_p in axes_flat:
            handles, labels = ax_p.get_legend_handles_labels()
            if handles: by_label = dict(zip(labels, handles)); ax_p.legend(by_label.values(), by_label.keys(), loc='best', fontsize=7)
        
        fig.suptitle(f"Comparación de Partes Individuales: {', '.join(flute_names_for_title)}", fontsize=11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._setup_plot_canvas(self.parts_frame, fig)

    def update_inharmonic_plot(self):
        if not self.acoustic_analysis_list_for_summary or not self.ordered_notes_for_summary: return
        fig = FluteOperations.plot_summary_cents_differences(
            self.acoustic_analysis_list_for_summary, 
            self.ordered_notes_for_summary
        )
        self._setup_plot_canvas(self.inharmonic_frame, fig)

    def update_moc_plot(self):
        if not self.acoustic_analysis_list_for_summary or not self.ordered_notes_for_summary or not self.finger_frequencies_map_for_summary: return
        fig = FluteOperations.plot_moc_summary(
            self.acoustic_analysis_list_for_summary, 
            self.finger_frequencies_map_for_summary,
            self.ordered_notes_for_summary
        )
        self._setup_plot_canvas(self.moc_frame, fig)

    def update_bi_espe_plot(self):
        if not self.acoustic_analysis_list_for_summary or not self.ordered_notes_for_summary or not self.finger_frequencies_map_for_summary: return
        fig = FluteOperations.plot_bi_espe_summary(
            self.acoustic_analysis_list_for_summary,
            self.finger_frequencies_map_for_summary,
            self.ordered_notes_for_summary
        )
        self._setup_plot_canvas(self.bi_espe_frame, fig)

    def update_admittance_note_options(self):
        if not self.ordered_notes_for_summary:
            self.note_combobox['values'] = []
            self.note_var.set("")
            for widget in self.admittance_plot_frame.winfo_children():
                widget.destroy()
            return
            
        self.note_combobox['values'] = self.ordered_notes_for_summary
        if self.ordered_notes_for_summary:
            self.note_var.set(self.ordered_notes_for_summary[0])
            self.update_admittance_plot(event=None)
        else:
            self.note_var.set("")

    def update_admittance_plot(self, event: Optional[tk.Event]):
        selected_note = self.note_var.get()
        if not selected_note or not self.acoustic_analysis_list_for_summary or not self.combined_measurements_list_for_summary:
            for widget in self.admittance_plot_frame.winfo_children():
                widget.destroy()
            # Podrías añadir un mensaje en el frame indicando que no hay datos.
            return
        
        fig = FluteOperations.plot_individual_admittance_analysis(
            self.acoustic_analysis_list_for_summary, 
            self.combined_measurements_list_for_summary,
            selected_note
        )
        self._setup_plot_canvas(self.admittance_plot_frame, fig)
    
    def open_json_editor(self):
        editor = TraditionalTextEditor(self)
        editor.grab_set()

    def close_app(self):
        if messagebox.askokcancel("Salir", "¿Está seguro de que desea salir de la aplicación?"):
            plt.close('all')
            self.destroy()

if __name__ == "__main__":
    # import logging
    # log_format = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s'
    # logging.basicConfig(level=logging.INFO, format=log_format) 
    # logging.getLogger('matplotlib').setLevel(logging.WARNING)
    # logger = logging.getLogger(__name__) 
    # logger.info("Iniciando la aplicación Flute Analyzer.")
        
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()
