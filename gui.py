import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle # Ya no se usa Circle para los agujeros en estos gráficos
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from flute_data import FluteData, FluteDataInitializationError # Asegúrate que FluteData se importa bien
from flute_operations import FluteOperations
from constants import BASE_COLORS, LINESTYLES, FLUTE_PARTS_ORDER
import logging # <--- AÑADIR ESTA LÍNEA

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_JSON_DIR = SCRIPT_DIR / "data_json"
if not DEFAULT_DATA_JSON_DIR.exists():
     DEFAULT_DATA_JSON_DIR = SCRIPT_DIR.parent / "data_json"

# --- Configuración básica de logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__) # Logger para este módulo (gui.py)

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
            logger.warning(f"ADVERTENCIA (Dialogo): El directorio de datos {current_data_dir_path} no es válido.")
            return
        try:
            sub_dir_names = [item_name for item_name in os.listdir(current_data_dir_path) if (current_data_dir_path / item_name).is_dir()]
            self.available_flute_paths = sorted(sub_dir_names)
            logger.debug(f"DEBUG (Dialogo): Flautas disponibles actualizadas: {self.available_flute_paths}")
        except OSError as e:
            logger.error(f"ERROR (Dialogo): Error listando directorio {current_data_dir_path}: {e.strerror}")

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
        self.selected_flute_dirs_on_accept = [] # No selection if cancelled
        self.final_data_dir_on_accept = self.current_data_dir # Keep current dir
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Análisis de Flautas")
        self.geometry("1200x800")

        self.data_dir = str(DEFAULT_DATA_JSON_DIR)
        self.flute_list_paths: List[str] = [] # This seems unused, consider removing
        self.currently_selected_flute_dirs: List[str] = []
        logger.debug(f"DEBUG: App.__init__ - ID(self): {id(self)}")
        logger.debug(f"DEBUG: App.__init__ - self.flute_list_paths: {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")

        style = ttk.Style(self) # Initialize style

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
        self.data_dir = dialog.final_data_dir_on_accept # Update data_dir based on dialog's final state
        if dialog.selected_flute_dirs_on_accept or (not dialog.selected_flute_dirs_on_accept and self.currently_selected_flute_dirs):
            # Load if new selection or if selection was cleared (to update plots)
            self.currently_selected_flute_dirs = dialog.selected_flute_dirs_on_accept
            self.load_flutes()
        # If dialog cancelled and no prior selection, do nothing more.

    def load_flutes(self):
        logger.debug(f"DEBUG: En load_flutes - ID(self): {id(self)}")
        logger.debug(f"DEBUG: En load_flutes - self.flute_list_paths (al inicio): {self.flute_list_paths}, ID: {id(self.flute_list_paths)}")

        selected_flute_dirs = self.currently_selected_flute_dirs
        logger.debug(f"DEBUG: En load_flutes - selected_flute_dirs: {selected_flute_dirs}")

        if not selected_flute_dirs:
            # Clear plots and data if no flutes are selected
            self.flute_ops_list = []
            self.acoustic_analysis_list_for_summary = []
            self.finger_frequencies_map_for_summary = {}
            self.combined_measurements_list_for_summary = []
            self.ordered_notes_for_summary = []
            self.loaded_flutes_label.config(text="Flautas cargadas: Ninguna")
            self.update_all_plots() # This will clear/placeholder the plots
            self.update_admittance_note_options() # Clear combobox
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
                    logger.debug(f"DEBUG: load_flutes - Intentando cargar FluteData desde: {data_path} (Intento en bucle while)")
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
                logger.debug(f"DEBUG: load_flutes - FluteData cargada para: {flute_dir_name}")
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
                logger.error(f"ERROR: load_flutes - attempt_successful_for_this_flute es True pero flute_data_obj es None para {flute_dir_name}")


        if not successful_loads and selected_flute_dirs:
             messagebox.showinfo("Carga Fallida", "No se pudo cargar ninguna de las flautas seleccionadas.")
             self.loaded_flutes_label.config(text="Flautas cargadas: Error en carga")
             return

        if successful_loads > 0:
            loaded_flutes_display_list = []
            for fo in self.flute_ops_list:
                flute_model_name = fo.flute_data.flute_model
                if not fo.flute_data.acoustic_analysis:
                    loaded_flutes_display_list.append(f"{flute_model_name} [Error Análisis Acústico]")
                else:
                    loaded_flutes_display_list.append(flute_model_name)
            loaded_names_str = ", ".join(loaded_flutes_display_list)
            self.loaded_flutes_label.config(text=f"Flautas cargadas: {loaded_names_str if loaded_names_str else 'Ninguna'}")
            logger.info(f"INFO: Se cargaron {successful_loads} flauta(s) exitosamente.")

            if self.finger_frequencies_map_for_summary:
                all_present_notes = set()
                for ff_map in self.finger_frequencies_map_for_summary.values():
                    all_present_notes.update(ff_map.keys())

                canonical_order = ["D", "D#", "E", "F", "Fs", "G", "G#", "A", "A#", "B", "C", "Cs"]
                self.ordered_notes_for_summary = [n for n in canonical_order if n in all_present_notes]
                self.ordered_notes_for_summary.extend(sorted(list(all_present_notes - set(self.ordered_notes_for_summary))))

            self.update_all_plots()
            self.update_admittance_note_options()
        else: # This case might be redundant if the first check for selected_flute_dirs handles it
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
        if not self.flute_ops_list: # If no flutes are loaded, clear/placeholder all plots
            self._setup_plot_canvas(self.profile_frame, plt.Figure()) # Placeholder for profile
            self._setup_plot_canvas(self.parts_frame, plt.Figure())   # Placeholder for parts
            self._setup_plot_canvas(self.admittance_plot_frame, plt.Figure()) # Placeholder for admittance
            self._setup_plot_canvas(self.inharmonic_frame, plt.Figure()) # Placeholder for inharmonicity
            self._setup_plot_canvas(self.moc_frame, plt.Figure()) # Placeholder for MOC
            self._setup_plot_canvas(self.bi_espe_frame, plt.Figure()) # Placeholder for BI/ESPE
            return
            
        self.update_profile_plot()
        self.update_parts_plot()
        self.update_inharmonic_plot()
        self.update_moc_plot()
        self.update_bi_espe_plot()

    def update_profile_plot(self):
        if not self.flute_ops_list:
            fig_placeholder, (ax_phys_ph, ax_acou_ph) = plt.subplots(2, 1)
            ax_phys_ph.text(0.5, 0.5, "Cargue flautas para ver el perfil físico.", ha='center', va='center', transform=ax_phys_ph.transAxes)
            ax_acou_ph.text(0.5, 0.5, "Cargue flautas para ver el perfil acústico.", ha='center', va='center', transform=ax_acou_ph.transAxes)
            self._setup_plot_canvas(self.profile_frame, fig_placeholder)
            return
        
        fig, (ax_physical, ax_acoustic) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        fig.subplots_adjust(hspace=0.3)

        # --- Subplot 1: Ensamblaje Físico Estimado ---
        flute_names_str_list_phys = [fo.flute_data.flute_model for fo in self.flute_ops_list]
        title_physical = f"Ensamblaje Físico Estimado: {', '.join(flute_names_str_list_phys)}"
        ax_physical.set_title(title_physical)
        ax_physical.set_xlabel("Posición Absoluta Estimada (mm)")
        ax_physical.set_ylabel("Diámetro (mm)")
        ax_physical.grid(True, linestyle=':', alpha=0.7)
        overall_max_x_physical_all_flutes = 0
        physical_legend_handles: List[plt.Line2D] = [] 

        for i, flute_ops in enumerate(self.flute_ops_list):
            flute_model_name = flute_ops.flute_data.flute_model
            max_x_this_flute = flute_ops.plot_physical_assembly(
                ax=ax_physical,
                plot_label_suffix="_nolegend_", 
                overall_linestyle=LINESTYLES[i % len(LINESTYLES)]
            )
            if max_x_this_flute is not None:
                 overall_max_x_physical_all_flutes = max(overall_max_x_physical_all_flutes, max_x_this_flute)
                 phys_line, = ax_physical.plot([], [], 
                                               color=BASE_COLORS[i % len(BASE_COLORS)],
                                               linestyle=LINESTYLES[i % len(LINESTYLES)], 
                                               label=f"{flute_model_name} (Físico: {max_x_this_flute:.1f} mm)")
                 physical_legend_handles.append(phys_line)

        if physical_legend_handles:
            ax_physical.legend(handles=physical_legend_handles, loc='best', fontsize='small')
        if overall_max_x_physical_all_flutes > 0:
            ax_physical.set_xlim(-10, overall_max_x_physical_all_flutes + 10)

        # --- Subplot 2: Perfil Acústico Interno Combinado ---
        flute_names_str_list_acou = [fo.flute_data.flute_model for fo in self.flute_ops_list]
        title_acoustic = f"Perfil Acústico Interno: {', '.join(flute_names_str_list_acou)}"
        ax_acoustic.set_title(title_acoustic)
        ax_acoustic.set_xlabel("Posición (mm) desde el corcho")
        ax_acoustic.set_ylabel("Diámetro (mm)")
        ax_acoustic.grid(True, linestyle=':', alpha=0.7)
        
        min_diam_all_acoustic_profiles = float('inf')
        max_overall_cork_relative_pos = -float('inf')
        min_overall_cork_relative_pos = float('inf')
        acoustic_legend_handles: List[plt.Line2D] = []

        for flute_ops_ac in self.flute_ops_list: 
            if flute_ops_ac.flute_data.combined_measurements:
                min_diam_this_flute = min(m['diameter'] for m in flute_ops_ac.flute_data.combined_measurements)
                min_diam_all_acoustic_profiles = min(min_diam_all_acoustic_profiles, min_diam_this_flute)
        
        for i, flute_ops in enumerate(self.flute_ops_list):
            flute_model_name = flute_ops.flute_data.flute_model
            headjoint_data_for_offset = flute_ops.flute_data.data.get(FLUTE_PARTS_ORDER[0], {})
            stopper_abs_pos_mm_for_offset = headjoint_data_for_offset.get('_calculated_stopper_absolute_position_mm', 0.0)
            
            acoustic_length_this_flute = 0.0
            combined_measurements_for_length = flute_ops.flute_data.combined_measurements
            if combined_measurements_for_length: 
                acoustic_start_abs = stopper_abs_pos_mm_for_offset 
                if combined_measurements_for_length: 
                    acoustic_end_abs = max(m['position'] for m in combined_measurements_for_length)
                    acoustic_length_this_flute = acoustic_end_abs - acoustic_start_abs

                    cork_rel_positions_this_flute = [(m['position'] - acoustic_start_abs) for m in combined_measurements_for_length]
                    if cork_rel_positions_this_flute:
                        max_overall_cork_relative_pos = max(max_overall_cork_relative_pos, max(cork_rel_positions_this_flute))
                        min_overall_cork_relative_pos = min(min_overall_cork_relative_pos, min(cork_rel_positions_this_flute))
            
            acoustic_line, = ax_acoustic.plot([], [],
                                              color=BASE_COLORS[i % len(BASE_COLORS)],
                                              linestyle=LINESTYLES[i % len(LINESTYLES)],
                                              label=f"{flute_model_name} (Acústico: {acoustic_length_this_flute:.1f} mm)")
            acoustic_legend_handles.append(acoustic_line)
            
            flute_ops.plot_combined_flute_data(
                ax=ax_acoustic,
                plot_label="_nolegend_", 
                flute_color=BASE_COLORS[i % len(BASE_COLORS)],
                flute_style=LINESTYLES[i % len(LINESTYLES)],
                show_mortise_markers=False, 
                x_axis_origin_offset=stopper_abs_pos_mm_for_offset
            )
            
            y_pos_holes_acoustic = (min_diam_all_acoustic_profiles if min_diam_all_acoustic_profiles != float('inf') else 10) - (3 + i * 1.5)
            part_physical_starts_map: Dict[str, float] = {}
            current_physical_connection_point_abs = 0.0
            stopper_abs_pos_mm = stopper_abs_pos_mm_for_offset

            for idx_part_calc, part_name_calc in enumerate(FLUTE_PARTS_ORDER):
                part_data_calc = flute_ops.flute_data.data.get(part_name_calc, {})
                part_total_length_calc = part_data_calc.get("Total length", 0.0)
                part_mortise_length_calc = part_data_calc.get("Mortise length", 0.0)
                if idx_part_calc == 0:
                    part_physical_starts_map[part_name_calc] = 0.0
                    current_physical_connection_point_abs = part_total_length_calc - part_mortise_length_calc
                elif idx_part_calc == 1:
                    part_physical_starts_map[part_name_calc] = current_physical_connection_point_abs
                    current_physical_connection_point_abs += part_total_length_calc
                else:
                    part_physical_starts_map[part_name_calc] = current_physical_connection_point_abs - part_mortise_length_calc
                    current_physical_connection_point_abs = part_physical_starts_map[part_name_calc] + part_total_length_calc

            for part_name_hole in FLUTE_PARTS_ORDER:
                part_data_hole = flute_ops.flute_data.data.get(part_name_hole, {})
                part_physical_start_abs_mm = part_physical_starts_map.get(part_name_hole, 0.0)
                for h_pos_rel, h_diam in zip(part_data_hole.get("Holes position", []), part_data_hole.get("Holes diameter", [])):
                    abs_physical_hole_pos = part_physical_start_abs_mm + h_pos_rel
                    plot_pos_on_acoustic = abs_physical_hole_pos - stopper_abs_pos_mm
                    marker_size_scaled = max(h_diam * 2.0, 4)
                    ax_acoustic.plot(plot_pos_on_acoustic, y_pos_holes_acoustic, marker='o', color=BASE_COLORS[i % len(BASE_COLORS)], markersize=marker_size_scaled, linestyle='None', alpha=0.7)

        if acoustic_legend_handles:
            ax_acoustic.legend(handles=acoustic_legend_handles, loc='best', fontsize='small')
        if max_overall_cork_relative_pos > -float('inf'):
            ax_acoustic.set_xlim(min_overall_cork_relative_pos - 10, max_overall_cork_relative_pos + 10)
        else:
            ax_acoustic.set_xlim(-50, 600)

        self._setup_plot_canvas(self.profile_frame, fig)

    def update_parts_plot(self):
        if not self.flute_ops_list:
            fig_placeholder, _ = plt.subplots(2,2) # Create a figure with 4 subplots
            for i, ax_ph_part in enumerate(fig_placeholder.axes):
                ax_ph_part.text(0.5, 0.5, f"Cargue flautas para ver Parte {i+1}", ha='center', va='center', transform=ax_ph_part.transAxes)
            self._setup_plot_canvas(self.parts_frame, fig_placeholder)
            return

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
                             color=current_flute_color, markersize=3, label=f"{flute_model_name}")

                part_data_dict = flute_ops_instance.flute_data.data.get(part_name, {})

                part_physical_total_length = part_data_dict.get("Total length", 0.0)
                part_mortise_length = part_data_dict.get("Mortise length", 0.0)
                part_acoustic_length = 0.0
                if part_name == FLUTE_PARTS_ORDER[0]: 
                    part_acoustic_length = part_physical_total_length - part_mortise_length
                elif part_name == FLUTE_PARTS_ORDER[1]: 
                    part_acoustic_length = part_physical_total_length
                else: 
                    part_acoustic_length = part_physical_total_length - part_mortise_length
                
                text_str = f"L. Total: {part_physical_total_length:.1f} mm\nL. Acústica: {part_acoustic_length:.1f} mm"
                ax_part.text(0.02, 0.98 - (flute_idx * 0.12), text_str, transform=ax_part.transAxes, 
                             ha='left', va='top', fontsize=6, color=current_flute_color,
                             bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='grey'))

                hole_positions_part = part_data_dict.get("Holes position", [])
                hole_diameters_part = part_data_dict.get("Holes diameter", [])

                if hole_positions_part and hole_diameters_part:
                    min_diam_this_part_this_flute = min(diameters) if diameters else 0
                    y_pos_for_holes = min_diam_this_part_this_flute - (5 + flute_idx * 1.5)

                    for h_pos, h_diam in zip(hole_positions_part, hole_diameters_part):
                        marker_size_scaled_part = max(h_diam * 2.0, 4) 
                        ax_part.plot(h_pos, y_pos_for_holes, marker='o', color=current_flute_color, markersize=marker_size_scaled_part, linestyle='None', alpha=0.7)

                ax_part.set_title(f"{part_name.capitalize()}", fontsize=9)
                ax_part.set_xlabel("Posición en parte (mm)", fontsize=8)
                ax_part.set_ylabel("Diámetro (mm)", fontsize=8)
                ax_part.grid(True, linestyle=':', alpha=0.5)
                ax_part.tick_params(axis='both', which='major', labelsize=7)

        for ax_p in axes_flat:
            handles, labels = ax_p.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax_p.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=7)

        fig.suptitle(f"Comparación de Partes Individuales: {', '.join(flute_names_for_title)}", fontsize=11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._setup_plot_canvas(self.parts_frame, fig)

    def update_inharmonic_plot(self):
        if not self.acoustic_analysis_list_for_summary or not self.ordered_notes_for_summary:
            self._setup_plot_canvas(self.inharmonic_frame, plt.Figure()) # Placeholder
            return
        fig = FluteOperations.plot_summary_cents_differences(
            self.acoustic_analysis_list_for_summary,
            self.ordered_notes_for_summary
        )
        self._setup_plot_canvas(self.inharmonic_frame, fig)

    def update_moc_plot(self):
        if not self.acoustic_analysis_list_for_summary or not self.ordered_notes_for_summary or not self.finger_frequencies_map_for_summary:
            self._setup_plot_canvas(self.moc_frame, plt.Figure()) # Placeholder
            return
        fig = FluteOperations.plot_moc_summary(
            self.acoustic_analysis_list_for_summary,
            self.finger_frequencies_map_for_summary,
            self.ordered_notes_for_summary
        )
        self._setup_plot_canvas(self.moc_frame, fig)

    def update_bi_espe_plot(self):
        if not self.acoustic_analysis_list_for_summary or not self.ordered_notes_for_summary or not self.finger_frequencies_map_for_summary:
            self._setup_plot_canvas(self.bi_espe_frame, plt.Figure()) # Placeholder
            return
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
            self._setup_plot_canvas(self.admittance_plot_frame, plt.Figure()) # Placeholder
            return

        self.note_combobox['values'] = self.ordered_notes_for_summary
        if self.ordered_notes_for_summary:
            self.note_var.set(self.ordered_notes_for_summary[0])
            self.update_admittance_plot(event=None)
        else:
            self.note_var.set("")
            self._setup_plot_canvas(self.admittance_plot_frame, plt.Figure()) # Placeholder

    def update_admittance_plot(self, event: Optional[tk.Event]):
        selected_note = self.note_var.get()
        if not selected_note or not self.acoustic_analysis_list_for_summary:
            self._setup_plot_canvas(self.admittance_plot_frame, plt.Figure()) # Placeholder
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
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.close_app)
    app.mainloop()
