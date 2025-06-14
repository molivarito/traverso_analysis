import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from flute_data import FluteData, FluteDataInitializationError
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
                messagebox.showinfo("Guardado", f"Archivo    guardado exitosamente:\n{self.filename}")
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

class FluteSelectionDialog(tk.Toplevel):
    def __init__(self, master, initial_data_dir, previously_selected_paths=None):
        super().__init__(master)
        self.title("Seleccionar Flautas")
        self.geometry("500x400")
        self.transient(master)
        self.grab_set()

        self.current_data_dir = initial_data_dir
        self.available_flute_paths: List[str] = []
        self.previously_selected_paths = previously_selected_paths if previously_selected_paths else []
        self.selected_flute_dirs_on_accept: List[str] = []
        self.final_data_dir_on_accept: str = initial_data_dir

        self.flute_checkbox_vars: Dict[str, tk.BooleanVar] = {}

        self._create_dialog_widgets()
        self._update_available_flute_paths_in_dialog()
        self._populate_flute_list()

        self.wait_window(self)

    def _create_dialog_widgets(self):
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)

        dir_selection_row = ttk.Frame(top_frame)
        dir_selection_row.pack(fill=tk.X, pady=(0,5))
        ttk.Label(dir_selection_row, text="Directorio:").pack(side=tk.LEFT, padx=(0,5))
        self.data_dir_label_dialog = ttk.Label(dir_selection_row, text=self.current_data_dir, relief="sunken", width=40)
        self.data_dir_label_dialog.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))
        browse_button_dialog = ttk.Button(dir_selection_row, text="Cambiar...", command=self._browse_data_directory_dialog)
        browse_button_dialog.pack(side=tk.LEFT)


        list_frame = ttk.LabelFrame(self, text="Flautas Disponibles", padding=10)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(list_frame, borderwidth=0, background="#ffffff")
        self.frame_checkboxes = ttk.Frame(self.canvas)
        self.scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas_frame_id = self.canvas.create_window((0, 0), window=self.frame_checkboxes, anchor="nw")

        self.frame_checkboxes.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(self.canvas_frame_id, width=e.width))

        button_frame = ttk.Frame(self, padding=10)
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="Aceptar", command=self._on_accept).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancelar", command=self._on_cancel).pack(side=tk.RIGHT)

    def _browse_data_directory_dialog(self):
        dir_path = filedialog.askdirectory(initialdir=self.current_data_dir, title="Seleccionar Directorio de Datos de Flautas", parent=self)
        if dir_path:
            self.current_data_dir = dir_path
            self.data_dir_label_dialog.config(text=self.current_data_dir)
            self._update_available_flute_paths_in_dialog()
            self._populate_flute_list()

    def _update_available_flute_paths_in_dialog(self):
        current_data_dir_path = Path(self.current_data_dir)
        self.available_flute_paths = []
        if not (current_data_dir_path.exists() and current_data_dir_path.is_dir()):
            print(f"ADVERTENCIA (Dialogo): El directorio de datos {current_data_dir_path} no es válido.")
            return
        try:
            sub_dir_names = [item_name for item_name in os.listdir(current_data_dir_path) if (current_data_dir_path / item_name).is_dir()]
            self.available_flute_paths = sorted(sub_dir_names)
            print(f"DEBUG (Dialogo): Flautas disponibles actualizadas: {self.available_flute_paths}")
        except OSError as e:
            print(f"ERROR (Dialogo): Error listando directorio {current_data_dir_path}: {e.strerror}")

    def _populate_flute_list(self):
        for widget in self.frame_checkboxes.winfo_children():
            widget.destroy()
        self.flute_checkbox_vars.clear()

        if not self.available_flute_paths:
            ttk.Label(self.frame_checkboxes, text="(No hay flautas en el directorio especificado)").pack(anchor="w")
        else:
            for flute_dir_name in self.available_flute_paths:
                var = tk.BooleanVar(value=(flute_dir_name in self.previously_selected_paths))
                cb = ttk.Checkbutton(self.frame_checkboxes, text=flute_dir_name, variable=var)
                cb.pack(anchor="w", padx=5, pady=1)
                self.flute_checkbox_vars[flute_dir_name] = var

    def _on_accept(self):
        self.selected_flute_dirs_on_accept = [name for name, var in self.flute_checkbox_vars.items() if var.get()]
        self.final_data_dir_on_accept = self.current_data_dir
        self.destroy()

    def _on_cancel(self):
        self.selected_flute_dirs_on_accept = []
        self.final_data_dir_on_accept = self.current_data_dir
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis de Flautas")
        self.geometry("1200x800")

        self.data_dir = str(DEFAULT_DATA_JSON_DIR)
        self.flute_list_paths: List[str] = []
        self.currently_selected_flute_dirs: List[str] = []
        print(f"DEBUG: App.__init__ - ID(self): {id(self)}")
        print(f"DEBUG: App.__init__ - BEFORE _update_available_flute_paths - self.flute_list_paths: {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")

        style = ttk.Style(self)

        top_bar = ttk.Frame(self)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10,5))
        exit_button = ttk.Button(top_bar, text="Salir de Aplicación", command=self.close_app)
        exit_button.pack(side=tk.RIGHT, padx=(5,0))

        editor_button_top_bar = ttk.Button(top_bar, text="Editor JSON", command=self.open_json_editor)
        editor_button_top_bar.pack(side=tk.RIGHT, padx=(0,5))

        config_frame = ttk.Frame(self, padding=(10,5))
        config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5,2))

        action_frame = ttk.Frame(config_frame)
        action_frame.pack(fill=tk.X, pady=(5,0))

        select_flutes_button = ttk.Button(action_frame, text="Seleccionar Flautas...", command=self.open_flute_selection_dialog)
        select_flutes_button.pack(side=tk.LEFT, padx=(0,10))

        self.loaded_flutes_label = ttk.Label(action_frame, text="Flautas cargadas: Ninguna", relief="groove", padding=(5,2), width=70)
        self.loaded_flutes_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,10))

        print(f"DEBUG: App.__init__ - AFTER _update_available_flute_paths - self.flute_list_paths: {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=(5,10))

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
        note_selection_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5,2))
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

    def open_flute_selection_dialog(self):
        dialog = FluteSelectionDialog(self, self.data_dir, self.currently_selected_flute_dirs)
        self.data_dir = dialog.final_data_dir_on_accept
        if dialog.selected_flute_dirs_on_accept:
            self.currently_selected_flute_dirs = dialog.selected_flute_dirs_on_accept
            self.load_flutes()
        else:
            self.currently_selected_flute_dirs = []
            self.loaded_flutes_label.config(text="Flautas cargadas: Ninguna")

    def load_flutes(self):
        print(f"DEBUG: En load_flutes - ID(self): {id(self)}")
        print(f"DEBUG: En load_flutes - self.flute_list_paths (al inicio): {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")

        selected_flute_dirs = self.currently_selected_flute_dirs
        print(f"DEBUG: En load_flutes - selected_flute_dirs (de checkboxes): {selected_flute_dirs}")

        if not selected_flute_dirs:
            messagebox.showwarning("Sin Selección", "Por favor, seleccione al menos una flauta.")
            self.loaded_flutes_label.config(text="Flautas cargadas: Ninguna")
            return

        self.flute_ops_list = []
        self.acoustic_analysis_list_for_summary = []
        self.finger_frequencies_map_for_summary = {}
        self.combined_measurements_list_for_summary = []
        self.ordered_notes_for_summary = []
        successful_loads = 0
        flute_data_obj: Optional[FluteData] = None 

        for flute_dir_name in selected_flute_dirs:
            data_path = Path(self.data_dir) / flute_dir_name
            attempt_successful_for_this_flute = False

            while True: 
                try:
                    print(f"DEBUG: load_flutes - Intentando cargar FluteData desde: {data_path} (Intento en bucle while)")
                    flute_data_obj_current_attempt = FluteData(str(data_path))

                    if not flute_data_obj_current_attempt.validation_errors:
                        if flute_data_obj_current_attempt.validation_warnings:
                            warning_messages = "\n".join([w.get('message', 'Advertencia desconocida.') for w in flute_data_obj_current_attempt.validation_warnings])
                            messagebox.showwarning("Advertencias de Validación", f"Advertencias para '{flute_dir_name}':\n{warning_messages}", parent=self)
                        
                        flute_data_obj = flute_data_obj_current_attempt
                        attempt_successful_for_this_flute = True
                        break 

                    error_info = flute_data_obj_current_attempt.validation_errors[0]
                    error_message = error_info.get('message', 'Error desconocido.')
                    part_with_error = error_info.get('part')
                    file_to_edit_path_obj: Optional[Path] = None
                    if part_with_error:
                        file_to_edit_path_obj = data_path / f"{part_with_error}.json"

                    prompt_message = f"Error en datos para '{flute_dir_name}':\n- {error_message}\n\n"
                    
                    if file_to_edit_path_obj and file_to_edit_path_obj.exists():
                        prompt_message += f"¿Desea editar el archivo '{file_to_edit_path_obj.name}' para corregirlo?"
                        user_choice = messagebox.askyesnocancel("Error de Datos", prompt_message, parent=self, icon=messagebox.ERROR)
                        if user_choice is True: 
                            editor = TraditionalTextEditor(self)
                            editor.filename = str(file_to_edit_path_obj)
                            try:
                                with open(file_to_edit_path_obj, "r", encoding="utf-8") as f_edit: content = f_edit.read()
                                editor.text.delete("1.0", tk.END); editor.text.insert(tk.END, content)
                                editor.title(f"Editando - {file_to_edit_path_obj.name}")
                                self.wait_window(editor)
                                continue 
                            except Exception as e_open_editor:
                                messagebox.showerror("Error Abriendo Editor", f"No se pudo abrir '{file_to_edit_path_obj.name}':\n{e_open_editor}", parent=self)
                                attempt_successful_for_this_flute = False
                                break 
                        elif user_choice is False: 
                            messagebox.showinfo("Carga Omitida", f"La flauta '{flute_dir_name}' no se cargará.", parent=self)
                            attempt_successful_for_this_flute = False
                            break 
                        else: 
                            messagebox.showinfo("Carga Cancelada", "Se canceló la carga de flautas.", parent=self)
                            self.flute_ops_list = []; self.currently_selected_flute_dirs = []
                            self.loaded_flutes_label.config(text="Flautas cargadas: Ninguna (cancelado)")
                            return 
                    else: 
                        messagebox.showerror("Error de Datos", f"Error en datos para '{flute_dir_name}':\n- {error_message}\n\nEsta flauta no se cargará.", parent=self)
                        attempt_successful_for_this_flute = False
                        break 
                
                except FluteDataInitializationError as e_fdi:
                    messagebox.showerror("Error de Carga (Procesamiento Interno)",
                                         f"Error al procesar datos para '{flute_dir_name}':\n{e_fdi}\n\nEsta flauta no se cargará.", parent=self)
                    attempt_successful_for_this_flute = False
                    break 
                
                except Exception as e_load_flute_data:
                    messagebox.showerror("Error de Carga (Inesperado)",
                                         f"Error inesperado al cargar datos para '{flute_dir_name}':\n{e_load_flute_data}\n\nEsta flauta no se cargará.", parent=self)
                    attempt_successful_for_this_flute = False
                    break 
            
            if not attempt_successful_for_this_flute:
                continue 

            if flute_data_obj: 
                print(f"DEBUG: load_flutes - FluteData cargada para: {flute_dir_name}")
                flute_ops_obj = FluteOperations(flute_data_obj)
                
                self.flute_ops_list.append(flute_ops_obj)
                flute_model_name = flute_data_obj.flute_model
                self.acoustic_analysis_list_for_summary.append(
                    (flute_data_obj.acoustic_analysis, flute_model_name)
                )
                self.combined_measurements_list_for_summary.append(
                    (flute_data_obj.combined_measurements, flute_model_name)
                )
                if flute_data_obj.finger_frequencies:
                    self.finger_frequencies_map_for_summary[flute_model_name] = flute_data_obj.finger_frequencies
                successful_loads += 1
            else: 
                print(f"ERROR: load_flutes - attempt_successful_for_this_flute es True pero flute_data_obj es None para {flute_dir_name}")


        if not successful_loads and selected_flute_dirs:
             messagebox.showinfo("Carga Fallida", "No se pudo cargar ninguna de las flautas seleccionadas.")
             self.loaded_flutes_label.config(text="Flautas cargadas: Error en carga")
             return

        if successful_loads > 0:
            loaded_flutes_display_list = []
            for fo in self.flute_ops_list:
                flute_model_name = fo.flute_data.flute_model
                # Comprobar si el análisis acústico está vacío o tiene algún problema
                # (asumiendo que un análisis vacío significa que falló para todas las notas)
                if not fo.flute_data.acoustic_analysis: # Chequea si el dict está vacío
                    loaded_flutes_display_list.append(f"{flute_model_name} [Error Análisis Acústico]")
                else:
                    loaded_flutes_display_list.append(flute_model_name)
            loaded_names_str = ", ".join(loaded_flutes_display_list)
            self.loaded_flutes_label.config(text=f"Flautas cargadas: {loaded_names_str if loaded_names_str else 'Ninguna'}")
            print(f"INFO: Se cargaron {successful_loads} flauta(s) exitosamente.")

            if self.finger_frequencies_map_for_summary:
                all_present_notes = set()
                for ff_map in self.finger_frequencies_map_for_summary.values():
                    all_present_notes.update(ff_map.keys())

                canonical_order = ["D", "D#", "E", "F", "Fs", "G", "G#", "A", "A#", "B", "C", "Cs"]
                self.ordered_notes_for_summary = [n for n in canonical_order if n in all_present_notes]
                self.ordered_notes_for_summary.extend(sorted(list(all_present_notes - set(self.ordered_notes_for_summary))))

            self.update_all_plots()
            self.update_admittance_note_options()
        else:
            self.loaded_flutes_label.config(text="Flautas cargadas: Ninguna (error en carga)")

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
            flute_model_name = flute_ops.flute_data.flute_model
            flute_ops.plot_combined_flute_data(
                ax=ax,
                plot_label=flute_model_name,
                flute_color=BASE_COLORS[i % len(BASE_COLORS)],
                flute_style=LINESTYLES[i % len(LINESTYLES)]
            )
            if flute_ops.flute_data.combined_measurements:
                min_diam_profile = min(m['diameter'] for m in flute_ops.flute_data.combined_measurements if 'diameter' in m) if flute_ops.flute_data.combined_measurements else 10
                y_pos_holes = min_diam_profile - (5 + i * 0.5)

                current_abs_pos_holes = 0.0
                for part_name_hole in FLUTE_PARTS_ORDER:
                    part_data_hole = flute_ops.flute_data.data.get(part_name_hole, {})
                    hole_positions_rel = part_data_hole.get("Holes position", [])
                    hole_diameters_rel = part_data_hole.get("Holes diameter", [])

                    for h_pos, h_diam in zip(hole_positions_rel, hole_diameters_rel):
                        abs_hole_pos = current_abs_pos_holes + h_pos
                        marker_size_scaled = max(h_diam * 0.8, 3)
                        ax.plot(abs_hole_pos, y_pos_holes,
                                marker='o', color=BASE_COLORS[i % len(BASE_COLORS)],
                                markersize=marker_size_scaled,
                                linestyle='None', alpha=0.7)

                    total_length_part = part_data_hole.get("Total length", 0.0)
                    mortise_length_part = part_data_hole.get("Mortise length", 0.0)
                    if FLUTE_PARTS_ORDER.index(part_name_hole) < len(FLUTE_PARTS_ORDER) -1 :
                        current_abs_pos_holes += (total_length_part - mortise_length_part)

        if len(self.flute_ops_list) > 0 :
            ax.legend(loc='best', title="Flautas")

        ax.set_title("Perfiles Combinados de Flautas")
        self._setup_plot_canvas(self.profile_frame, fig)

    def update_parts_plot(self):
        if not self.flute_ops_list: return

        fig, axes_array = plt.subplots(2, 2, figsize=(12, 10))
        axes_flat = list(axes_array.flatten())

        flute_names_for_title = []

        for flute_idx, flute_ops_instance in enumerate(self.flute_ops_list):
            flute_model_name = flute_ops_instance.flute_data.flute_model
            if flute_model_name not in flute_names_for_title:
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
                        marker_size_scaled_part = max(h_diam * 0.8, 3)
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
        if not selected_note or not self.acoustic_analysis_list_for_summary:
            for widget in self.admittance_plot_frame.winfo_children():
                widget.destroy()
            return

        fig = FluteOperations.plot_individual_admittance_analysis(
            self.acoustic_analysis_list_for_summary,
            self.combined_measurements_list_for_summary, # Añadido para que coincida con la firma esperada
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
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()
