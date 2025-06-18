import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from matplotlib.patches import Circle # No se usa directamente aquí
import numpy as np # Para np.nan si es necesario en algún placeholder
import json
import os
import copy
from pathlib import Path # <--- AÑADIR ESTA LÍNEA
from typing import Dict, Any, Optional, List, Tuple # <--- AÑADIR ESTA LÍNEA
from flute_data import FluteData, FluteDataInitializationError # <--- MODIFICAR ESTA LÍNEA
from flute_operations import FluteOperations
# Asumiendo que GraphicalFluteEditor existe y funciona como se espera
# Necesitarás asegurarte de que este import funcione en tu estructura de proyecto.
# Si está en el mismo directorio, esto debería estar bien.
from graphical_editor import GraphicalFluteEditor
from constants import FLUTE_PARTS_ORDER # <--- LÍNEA AÑADIDA

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# --- Default Paths ---
SCRIPT_DIR_EXPERIMENTER = Path(__file__).resolve().parent
DEFAULT_DATA_DIR_EXPERIMENTER = SCRIPT_DIR_EXPERIMENTER.parent / "data_json"
if not DEFAULT_DATA_DIR_EXPERIMENTER.exists():
    DEFAULT_DATA_DIR_EXPERIMENTER = SCRIPT_DIR_EXPERIMENTER / "data_json" # Fallback if structure is different

# Definición de TraditionalTextEditor (adaptada de gui.py para ser autocontenida aquí)
class TraditionalTextEditor(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Editor JSON") # Título más genérico
        self.geometry("800x600")
        self.filename = None # Ruta al archivo que se está editando
        self.create_widgets()

    def create_widgets(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        # No necesitamos 'Abrir' o 'Guardar Como' si el editor se usa solo para corregir un archivo específico.
        btn_save = ttk.Button(toolbar, text="Guardar y Cerrar", command=self.save_and_close)
        btn_save.pack(side=tk.LEFT, padx=5, pady=2)
        btn_cancel = ttk.Button(toolbar, text="Cancelar", command=self.cancel_edit)
        btn_cancel.pack(side=tk.LEFT, padx=5, pady=2)

        self.text_area = tk.Text(self, wrap=tk.WORD, undo=True)
        self.text_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.text_area.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.configure(yscrollcommand=vsb.set)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text_area.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_area.configure(xscrollcommand=hsb.set)

    def load_file_content(self, filepath: str):
        self.filename = filepath
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, content)
            self.title(f"Editando - {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error Abriendo Archivo", f"No se pudo cargar el archivo para editar:\n{e}", parent=self)
            self.destroy() # Cerrar si no se puede cargar

    def save_and_close(self):
        if self.filename:
            try:
                with open(self.filename, "w", encoding="utf-8") as f:
                    f.write(self.text_area.get("1.0", tk.END).strip() + "\n") # Asegurar una nueva línea al final
                logger.info(f"Archivo guardado: {self.filename}")
            except Exception as e:
                messagebox.showerror("Error Guardando", f"No se pudo guardar el archivo:\n{e}", parent=self)
                return # No cerrar si falla el guardado
        self.destroy()

    def cancel_edit(self):
        self.destroy()

class FluteExperimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Flute Geometry Experimenter")
        self.geometry("1250x900") # Un poco más grande

        # --- Datos ---
        self.original_flute_data_dict: Optional[Dict[str, Any]] = None # Datos JSON originales
        self.modified_flute_data_dict: Optional[Dict[str, Any]] = None # Datos JSON modificados

        self.original_flute_ops: Optional[FluteOperations] = None # Para análisis original
        self.modified_flute_ops: Optional[FluteOperations] = None # Para análisis modificado

        self.flute_name: str = ""
        self.data_path: str = "" # Ruta al directorio original cargado
        self.has_modifications: bool = False # Si modified_flute_data_dict difiere del original
        self.data_modified_since_analysis: bool = False # Si modified_flute_data_dict cambió desde el último análisis

        self.create_menu()
        self.create_main_layout()
        self._configure_plot_axes_placeholders()

        self.protocol("WM_DELETE_WINDOW", self._on_close_app)

    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Flute Directory...", command=self._load_flute_from_dialog)
        file_menu.add_command(label="Save Modified Geometry As...", command=self._save_modified_as, state=tk.DISABLED)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close_app)
        self.file_menu = file_menu # Guardar referencia para habilitar/deshabilitar items

    def create_main_layout(self):
        control_frame = ttk.Frame(self, padding="5")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.load_button = ttk.Button(control_frame, text="Load Flute...", command=self._load_flute_from_dialog)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.edit_button = ttk.Button(control_frame, text="Edit Geometry", command=self.open_geometry_editor, state=tk.DISABLED)
        self.edit_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.reset_button = ttk.Button(control_frame, text="Reset to Original", command=self._reset_modifications, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.analyze_button = ttk.Button(control_frame, text="Analyze Modified", command=self._analyze_modified, state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.flute_name_label = ttk.Label(control_frame, text="No flute loaded.", font=('Arial', 10, 'italic'), width=40, anchor="e")
        self.flute_name_label.pack(side=tk.RIGHT, padx=10, fill=tk.X, expand=True)

        plot_main_frame = ttk.Frame(self, padding="5")
        plot_main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        plot_main_frame.columnconfigure(0, weight=1); plot_main_frame.columnconfigure(1, weight=1)
        plot_main_frame.rowconfigure(0, weight=1); plot_main_frame.rowconfigure(1, weight=1)

        self.frame_geom = ttk.Frame(plot_main_frame, borderwidth=1, relief=tk.SUNKEN)
        self.frame_inharmonic = ttk.Frame(plot_main_frame, borderwidth=1, relief=tk.SUNKEN)
        self.frame_moc = ttk.Frame(plot_main_frame, borderwidth=1, relief=tk.SUNKEN)
        self.frame_bi_espe = ttk.Frame(plot_main_frame, borderwidth=1, relief=tk.SUNKEN)

        self.frame_geom.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.frame_inharmonic.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.frame_moc.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)
        self.frame_bi_espe.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

        self.fig_geom, self.ax_geom = plt.subplots()
        self.fig_inharmonic, self.ax_inharmonic = plt.subplots()
        self.fig_moc, self.ax_moc = plt.subplots()
        self.fig_bi_espe, self.ax_bi_espe = plt.subplots()

        self.canvas_geom = self._create_canvas(self.fig_geom, self.frame_geom)
        self.canvas_inharmonic = self._create_canvas(self.fig_inharmonic, self.frame_inharmonic)
        self.canvas_moc = self._create_canvas(self.fig_moc, self.frame_moc)
        self.canvas_bi_espe = self._create_canvas(self.fig_bi_espe, self.frame_bi_espe)

    def _create_canvas(self, fig, parent_frame):
        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        return canvas

    def _configure_plot_axes_placeholders(self, specific_ax=None):
        axes_map = {
            "geom": (self.ax_geom, "Flute Geometry Profile"),
            "inharmonic": (self.ax_inharmonic, "Inharmonicity (F2 vs 2*F1)"),
            "moc": (self.ax_moc, "Modal Octave Compression (MOC)"),
            "bi_espe": (self.ax_bi_espe, "B_I / ESPE")
        }
        canvases_map = {
            "geom": self.canvas_geom, "inharmonic": self.canvas_inharmonic,
            "moc": self.canvas_moc, "bi_espe": self.canvas_bi_espe
        }

        axes_to_configure = []
        if specific_ax:
            for key, (ax_obj, title) in axes_map.items():
                if ax_obj == specific_ax:
                    axes_to_configure.append((ax_obj, title, canvases_map[key]))
                    break
        else:
            for key, (ax_obj, title) in axes_map.items():
                axes_to_configure.append((ax_obj, title, canvases_map[key]))

        for ax, title, canvas_to_draw in axes_to_configure:
            ax.clear()
            ax.set_title(title, fontsize=10)
            ax.text(0.5, 0.5, "Load Flute Data", ha='center', va='center', transform=ax.transAxes, fontsize=12, color='grey')
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
            canvas_to_draw.draw_idle()

    def _load_flute_from_dialog(self):
        initial_dir = str(DEFAULT_DATA_DIR_EXPERIMENTER) if DEFAULT_DATA_DIR_EXPERIMENTER.exists() else "."
        dir_path = filedialog.askdirectory(title="Select Flute Data Directory", initialdir=initial_dir)
        if not dir_path or not os.path.isdir(dir_path):
            logger.info("Flute loading cancelled or invalid path.")
            return
        self._process_loaded_flute_data(dir_path)
        
    def _process_loaded_flute_data(self, dir_path_str: str):
        dir_path = Path(dir_path_str)
        base_name = dir_path.name
        flute_data_obj_for_experimenter: Optional[FluteData] = None

        while True: # Bucle para reintentar después de editar
            try:
                logger.info(f"Experimenter: Intentando cargar FluteData desde: {dir_path}")
                # FluteData se encarga de leer los JSONs y la validación inicial.
                # Pasamos skip_acoustic_analysis=True porque solo necesitamos la geometría
                # para la carga inicial y edición. El análisis se hará explícitamente.
                current_flute_data_attempt = FluteData(str(dir_path), 
                                                       source_name=base_name,
                                                       skip_acoustic_analysis=True)

                if not current_flute_data_attempt.validation_errors:
                    if current_flute_data_attempt.validation_warnings:
                        warning_messages = "\n".join([w.get('message', 'Advertencia desconocida.') for w in current_flute_data_attempt.validation_warnings])
                        messagebox.showwarning("Advertencias de Validación", f"Advertencias para '{base_name}':\n{warning_messages}", parent=self)
                    flute_data_obj_for_experimenter = current_flute_data_attempt
                    break # Carga exitosa, salir del bucle while

                # Hay errores de validación
                error_info = current_flute_data_attempt.validation_errors[0]
                error_message = error_info.get('message', 'Error desconocido.')
                part_with_error = error_info.get('part')
                file_to_edit_path_obj: Optional[Path] = None
                if part_with_error:
                    file_to_edit_path_obj = dir_path / f"{part_with_error}.json"

                prompt_message = f"Error en datos JSON para '{base_name}':\n- {error_message}\n\n"
                
                if file_to_edit_path_obj and file_to_edit_path_obj.exists():
                    prompt_message += f"¿Desea editar el archivo '{file_to_edit_path_obj.name}' para corregirlo?"
                    user_choice = messagebox.askyesnocancel("Error de Datos JSON", prompt_message, parent=self, icon=messagebox.ERROR)
                    if user_choice is True: # Sí, editar
                        editor = TraditionalTextEditor(self)
                        editor.load_file_content(str(file_to_edit_path_obj))
                        self.wait_window(editor) # Esperar a que el editor se cierre
                        continue # Reintentar carga en el bucle while
                    elif user_choice is False: # No, no editar
                        flute_data_obj_for_experimenter = None; break 
                    else: # Cancelar
                        messagebox.showinfo("Carga Cancelada", "Se canceló la carga de la flauta.", parent=self)
                        self._reset_state_after_load_fail(); return
                else: # Error general o archivo no identificable
                    messagebox.showerror("Error de Datos JSON", prompt_message, parent=self)
                    flute_data_obj_for_experimenter = None; break
            
            except FluteDataInitializationError as e_fdi: # Errores más allá de la validación JSON básica
                messagebox.showerror("Error de Carga (Procesamiento)", f"Error al procesar datos para '{base_name}':\n{e_fdi}", parent=self)
                flute_data_obj_for_experimenter = None; break
            except Exception as e_load_fd: # Otros errores inesperados durante la instanciación de FluteData
                logger.exception(f"Error inesperado al cargar FluteData para {base_name}")
                messagebox.showerror("Error de Carga (Inesperado)", f"Error inesperado al cargar '{base_name}':\n{e_load_fd}", parent=self)
                flute_data_obj_for_experimenter = None; break
        
        if not flute_data_obj_for_experimenter:
            self._reset_state_after_load_fail()
            return

        # Si la carga fue exitosa (con o sin correcciones)
        try:
            self.data_path = str(dir_path)
            self.flute_name = base_name
            # Usar los datos directamente del objeto FluteData cargado
            self.original_flute_data_dict = copy.deepcopy(flute_data_obj_for_experimenter.data)
            self.modified_flute_data_dict = copy.deepcopy(self.original_flute_data_dict)
            self.has_modifications = False
            self.data_modified_since_analysis = True

            logger.info(f"Creating FluteData for original: {self.flute_name}")
            # Re-instanciar FluteData para el original_flute_ops, esta vez permitiendo el análisis acústico
            original_fd_for_ops = FluteData(source=copy.deepcopy(self.original_flute_data_dict), source_name=self.flute_name, skip_acoustic_analysis=False)
            if original_fd_for_ops.validation_errors: # Comprobar de nuevo por si acaso
                messagebox.showerror("Error Post-Carga", f"Errores de validación al re-procesar {self.flute_name} para análisis:\n{original_fd_for_ops.validation_errors[0]['message']}", parent=self)
                self._reset_state_after_load_fail(); return
            self.original_flute_ops = FluteOperations(original_fd_for_ops)
            logger.info("Original analysis complete.")

            self.modified_flute_ops = None

            self.file_menu.entryconfig("Save Modified Geometry As...", state=tk.NORMAL)
            self.edit_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.DISABLED)
            self.analyze_button.config(state=tk.DISABLED)
            self.title(f"Flute Experimenter - {self.flute_name}")
            self.flute_name_label.config(text=f"Loaded: {self.flute_name}")

            self._update_all_plots()

        except Exception as e:
            logger.exception("Error processing loaded flute data after successful FluteData instantiation")
            messagebox.showerror("Processing Error", f"Error after loading flute data:\n{type(e).__name__}: {e}", parent=self)
            self._reset_state_after_load_fail()

    def _reset_state_after_load_fail(self):
        self.original_flute_data_dict = None; self.modified_flute_data_dict = None
        self.original_flute_ops = None; self.modified_flute_ops = None
        self.flute_name = ""; self.data_path = ""; self.has_modifications = False
        self.data_modified_since_analysis = False
        self.file_menu.entryconfig("Save Modified Geometry As...", state=tk.DISABLED)
        self.edit_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.DISABLED)
        self.title("Flute Experimenter"); self.flute_name_label.config(text="No flute loaded.")
        self._configure_plot_axes_placeholders()

    def open_geometry_editor(self):
        if not self.modified_flute_data_dict:
            messagebox.showerror("Error", "No flute data loaded to edit.", parent=self)
            return
        editor_data_copy = copy.deepcopy(self.modified_flute_data_dict)
        # Asumiendo que GraphicalFluteEditor se importa correctamente
        editor = GraphicalFluteEditor(self, editor_data_copy, self.flute_name, self._editor_applied_callback)
        # editor.grab_set() # Descomentar si se quiere que el editor sea modal

    def _editor_applied_callback(self, updated_data_from_editor: Dict[str, Any]):
        logger.info("Received updated data from graphical editor.")
        if updated_data_from_editor != self.modified_flute_data_dict:
            self.modified_flute_data_dict = updated_data_from_editor
            self.has_modifications = True
            self.data_modified_since_analysis = True
            self.analyze_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.file_menu.entryconfig("Save Modified Geometry As...", state=tk.NORMAL)
            logger.info("Modifications applied from editor.")
        else:
            logger.info("Editor closed, but data is identical to current modified data.")

        self._update_geometry_plot()

    def _reset_modifications(self):
        if not self.original_flute_data_dict:
            messagebox.showinfo("Info", "No original data loaded to reset to.", parent=self)
            return
        if not self.has_modifications: # Solo resetear si hay modificaciones
            messagebox.showinfo("Info", "No modifications to reset.", parent=self)
            return

        if messagebox.askyesno("Reset Modifications",
                               "Revert all changes to the original loaded geometry?\n"
                               "This will clear any unsaved modifications.", parent=self):
            self.modified_flute_data_dict = copy.deepcopy(self.original_flute_data_dict)
            self.modified_flute_ops = None
            self.has_modifications = False
            self.data_modified_since_analysis = True # El estado "modificado" ahora es el original, necesita reanálisis si se quiere "analizar modificado"
            
            self.analyze_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            # Save As sigue activo porque podrías querer guardar el original (ahora en modified_flute_data_dict) con otro nombre
            
            logger.info("Modifications have been reset to original.")
            self._update_all_plots() # Esto mostrará solo el original ya que modified_flute_ops es None


    def _analyze_modified(self):
        if not self.modified_flute_data_dict:
            messagebox.showinfo("Info", "No modified data to analyze.", parent=self)
            return
        if not self.has_modifications:
             messagebox.showinfo("Info", "No modifications detected from original to analyze.", parent=self)
             return

        if not self.data_modified_since_analysis:
             if not messagebox.askyesno("Re-Analyze?",
                                        "Current modified geometry has already been analyzed. Re-analyze anyway?",
                                        parent=self):
                  return

        logger.info(f"Analyzing modified geometry for {self.flute_name}...")
        try:
            # Asegurar orden de mediciones y agujeros antes del análisis
            # Esta es una copia profunda, así que podemos modificarla sin afectar self.modified_flute_data_dict directamente aquí
            data_to_analyze = copy.deepcopy(self.modified_flute_data_dict)
            from constants import FLUTE_PARTS_ORDER # Asegurar importación
            for part_name in FLUTE_PARTS_ORDER:
                part_data = data_to_analyze.get(part_name) # Usar .get para seguridad
                if isinstance(part_data, dict):
                    if 'measurements' in part_data and isinstance(part_data['measurements'], list):
                        part_data['measurements'].sort(key=lambda item: item.get('position', 0.0))

                    if 'Holes position' in part_data and isinstance(part_data['Holes position'], list) and \
                       'Holes diameter' in part_data and isinstance(part_data['Holes diameter'], list):
                        pos_list = part_data['Holes position']
                        diam_list = part_data['Holes diameter']
                        if len(pos_list) == len(diam_list) and len(pos_list) > 0:
                            try:
                                valid_hole_data = []
                                for p, d in zip(pos_list, diam_list):
                                    if p is not None and d is not None:
                                        try: valid_hole_data.append((float(p), float(d)))
                                        except (ValueError, TypeError):
                                            logger.warning(f"Invalid hole data p={p}, d={d} in part {part_name}")
                                if valid_hole_data:
                                    sorted_holes = sorted(valid_hole_data, key=lambda pair: pair[0])
                                    part_data['Holes position'] = [p_val for p_val,d_val in sorted_holes]
                                    part_data['Holes diameter'] = [d_val for p_val,d_val in sorted_holes]
                            except Exception as e_sort:
                                logger.warning(f"Could not sort holes for part {part_name} during analysis: {e_sort}")

            mod_ops_name = f"{self.flute_name}_mod" if self.flute_name else "ModifiedFlute"
            # Pasar la copia data_to_analyze, que ya tiene el "Flute Model" del original
            # o se puede establecer explícitamente. FluteData lo manejará.
            if "Flute Model" not in data_to_analyze: # Asegurar que el dict tiene el nombre del modelo
                data_to_analyze["Flute Model"] = mod_ops_name

            temp_mod_flute_data_obj = FluteData(source=data_to_analyze, source_name=mod_ops_name)
            self.modified_flute_ops = FluteOperations(temp_mod_flute_data_obj)
            logger.info("Modified geometry analysis complete.")

            self.data_modified_since_analysis = False
            self._update_all_plots()

        except Exception as e:
            logger.exception("Error analyzing modified geometry")
            messagebox.showerror("Analysis Error", f"Could not analyze modified geometry:\n{type(e).__name__}: {e}", parent=self)
            self.modified_flute_ops = None
            self._update_all_plots()


    def _update_all_plots(self):
        logger.debug("Updating all plots...")
        self._update_geometry_plot()
        self._update_acoustic_plots()

    def _update_geometry_plot(self):
        logger.debug("Updating geometry plot...")
        ax = self.ax_geom
        ax.clear()
        plot_success = False
        legend_handles = []

        overall_max_x_physical_all_flutes = 0 # Para ajustar el límite X del gráfico

        if self.original_flute_ops:
            try:
                # Usar plot_physical_assembly para dibujar el ensamblaje físico
                max_x_orig = self.original_flute_ops.plot_physical_assembly(
                    ax=ax, 
                    plot_label_suffix=f"{self.original_flute_ops.flute_data.flute_model} (Original)",
                    overall_linestyle='-' # Usar el color por defecto de la función, solo especificar estilo
                )
                if max_x_orig is not None:
                    overall_max_x_physical_all_flutes = max(overall_max_x_physical_all_flutes, max_x_orig)
                plot_success = True

                # Dibujar agujeros para el original
                all_physical_diameters_orig = []
                for part_n_orig in FLUTE_PARTS_ORDER:
                    part_d_orig = self.original_flute_ops.flute_data.data.get(part_n_orig, {})
                    measurements_orig = part_d_orig.get("measurements", [])
                    if measurements_orig:
                        all_physical_diameters_orig.extend([m['diameter'] for m in measurements_orig if 'diameter' in m])
                
                min_overall_physical_diameter_orig = min(all_physical_diameters_orig) if all_physical_diameters_orig else 10
                y_pos_holes_orig = min_overall_physical_diameter_orig - 5 

                # Lógica para calcular el inicio físico de cada parte (para posicionar agujeros)
                part_physical_starts_map_orig: Dict[str, float] = {}
                current_physical_connection_point_abs_orig = 0.0
                # No necesitamos stopper_abs_pos_mm_orig para el ensamblaje físico

                for idx_part_calc, part_name_calc in enumerate(FLUTE_PARTS_ORDER):
                    part_data_calc = self.original_flute_ops.flute_data.data.get(part_name_calc, {})
                    part_total_length_calc = part_data_calc.get("Total length", 0.0)
                    part_mortise_length_calc = part_data_calc.get("Mortise length", 0.0)

                    if idx_part_calc == 0: # Headjoint
                        part_physical_starts_map_orig[part_name_calc] = 0.0
                        current_physical_connection_point_abs_orig = part_total_length_calc - part_mortise_length_calc
                    elif idx_part_calc == 1: # Left
                        part_physical_starts_map_orig[part_name_calc] = current_physical_connection_point_abs_orig
                        current_physical_connection_point_abs_orig += part_total_length_calc
                    else: # Right, Foot
                        part_physical_starts_map_orig[part_name_calc] = current_physical_connection_point_abs_orig - part_mortise_length_calc
                        current_physical_connection_point_abs_orig = part_physical_starts_map_orig[part_name_calc] + part_total_length_calc
                
                for part_name_hole_orig in FLUTE_PARTS_ORDER:
                    part_data_hole_orig = self.original_flute_ops.flute_data.data.get(part_name_hole_orig, {})
                    part_physical_start_abs_mm_orig = part_physical_starts_map_orig.get(part_name_hole_orig, 0.0)
                    hole_positions_orig = part_data_hole_orig.get("Holes position", [])
                    hole_diameters_orig = part_data_hole_orig.get("Holes diameter", [])
                    
                    for h_pos_rel, h_diam in zip(hole_positions_orig, hole_diameters_orig):
                        abs_physical_hole_pos = part_physical_start_abs_mm_orig + h_pos_rel
                        plot_pos_on_physical_assembly = abs_physical_hole_pos # Ya es la posición física absoluta
                        
                        # Usar 'o' como marcador y escalar markersize con el diámetro del agujero
                        marker_size_scaled = max(h_diam * 0.8, 3) 
                        ax.plot(plot_pos_on_physical_assembly, y_pos_holes_orig, marker='o', 
                                color='blue', markersize=marker_size_scaled, 
                                linestyle='None', alpha=0.6)
                # La leyenda de las partes ya la maneja plot_physical_assembly
                # Los agujeros no necesitan leyenda propia si se colorean igual que la flauta.

            except Exception as e: logger.error(f"Error plotting original geometry: {e}")

        if self.has_modifications and self.modified_flute_data_dict:
            logger.debug("Plotting modified geometry from current data dictionary for comparison.")
            try:
                # Crear FluteData temporalmente para graficar el estado actual de modified_flute_data_dict
                # Esto no reemplaza self.modified_flute_ops, que solo se crea tras un análisis explícito.
                mod_display_name = f"{self.flute_name}_mod_preview"
                temp_mod_data_obj = FluteData(source=copy.deepcopy(self.modified_flute_data_dict), source_name=mod_display_name)
                temp_mod_ops = FluteOperations(temp_mod_data_obj) # FluteOperations para la modificada

                max_x_mod = temp_mod_ops.plot_physical_assembly(
                    ax=ax,
                    plot_label_suffix=f"{self.flute_name} (Modificada)",
                    overall_linestyle='--' # Usar el color por defecto, solo especificar estilo
                )
                if max_x_mod is not None:
                    overall_max_x_physical_all_flutes = max(overall_max_x_physical_all_flutes, max_x_mod)

                # Dibujar agujeros para el modificado
                all_physical_diameters_mod = []
                for part_n_mod in FLUTE_PARTS_ORDER:
                    part_d_mod = temp_mod_ops.flute_data.data.get(part_n_mod, {})
                    measurements_mod = part_d_mod.get("measurements", [])
                    if measurements_mod:
                        all_physical_diameters_mod.extend([m['diameter'] for m in measurements_mod if 'diameter' in m])
                
                min_overall_physical_diameter_mod = min(all_physical_diameters_mod) if all_physical_diameters_mod else 10
                y_pos_holes_mod = min_overall_physical_diameter_mod - 7 # Un poco más abajo para diferenciar

                part_physical_starts_map_mod: Dict[str, float] = {}
                current_physical_connection_point_abs_mod = 0.0
                # No necesitamos stopper_abs_pos_mm_mod para el ensamblaje físico

                for idx_part_calc_mod, part_name_calc_mod in enumerate(FLUTE_PARTS_ORDER):
                    part_data_calc_mod = temp_mod_ops.flute_data.data.get(part_name_calc_mod, {})
                    part_total_length_calc_mod = part_data_calc_mod.get("Total length", 0.0)
                    part_mortise_length_calc_mod = part_data_calc_mod.get("Mortise length", 0.0)

                    if idx_part_calc_mod == 0: # Headjoint
                        part_physical_starts_map_mod[part_name_calc_mod] = 0.0
                        current_physical_connection_point_abs_mod = part_total_length_calc_mod - part_mortise_length_calc_mod
                    elif idx_part_calc_mod == 1: # Left
                        part_physical_starts_map_mod[part_name_calc_mod] = current_physical_connection_point_abs_mod
                        current_physical_connection_point_abs_mod += part_total_length_calc_mod
                    else: # Right, Foot
                        part_physical_starts_map_mod[part_name_calc_mod] = current_physical_connection_point_abs_mod - part_mortise_length_calc_mod
                        current_physical_connection_point_abs_mod = part_physical_starts_map_mod[part_name_calc_mod] + part_total_length_calc_mod

                for part_name_hole_mod in FLUTE_PARTS_ORDER:
                    part_data_hole_mod = temp_mod_ops.flute_data.data.get(part_name_hole_mod, {})
                    part_physical_start_abs_mm_mod = part_physical_starts_map_mod.get(part_name_hole_mod, 0.0)
                    hole_positions_mod = part_data_hole_mod.get("Holes position", [])
                    hole_diameters_mod = part_data_hole_mod.get("Holes diameter", [])

                    for h_pos_rel_m, h_diam_m in zip(hole_positions_mod, hole_diameters_mod):
                        abs_physical_hole_pos_m = part_physical_start_abs_mm_mod + h_pos_rel_m
                        plot_pos_on_physical_assembly_m = abs_physical_hole_pos_m # Ya es la posición física absoluta
                        # Usar 'o' como marcador también para los modificados, el color ya los diferencia
                        marker_size_scaled_m = max(h_diam_m * 0.8, 3)
                        ax.plot(plot_pos_on_physical_assembly_m, y_pos_holes_mod, marker='o', 
                                color='orange', markersize=marker_size_scaled_m, 
                                linestyle='None', alpha=0.6)

                plot_success = True
            except Exception as e: logger.error(f"Error plotting modified geometry from dict: {e}")

        if plot_success:
            # El título y las etiquetas de los ejes ya se establecen dentro de plot_physical_assembly
            # Si se llama varias veces, se sobrescribirán, lo cual está bien.
            # ax.set_title(f"Ensamblaje Físico Estimado: {self.flute_name}", fontsize=10)
            # ax.set_xlabel("Posición Absoluta Estimada (mm)"); ax.set_ylabel("Diámetro (mm)")
            ax.grid(True, linestyle=':', alpha=0.7)
            handles_all, labels_all = ax.get_legend_handles_labels()
            by_label_all = dict(zip(labels_all, handles_all)) # Eliminar duplicados de leyenda
            ax.legend(by_label_all.values(), by_label_all.keys(), loc='best', fontsize=9)
            if overall_max_x_physical_all_flutes > 0:
                ax.set_xlim(-10, overall_max_x_physical_all_flutes + 10)
        else:
            self._configure_plot_axes_placeholders(specific_ax=ax)
        self.canvas_geom.draw_idle()

    def _update_acoustic_plots(self):
        logger.debug("Updating acoustic plots...")
        if not self.original_flute_ops:
             self._configure_plot_axes_placeholders(self.ax_inharmonic)
             self._configure_plot_axes_placeholders(self.ax_moc)
             self._configure_plot_axes_placeholders(self.ax_bi_espe)
             return

        ops_orig = self.original_flute_ops
        ops_mod = self.modified_flute_ops # Puede ser None

        analysis_list_for_plot: List[Tuple[Dict[str,Any], str]] = []
        finger_freqs_map_for_plot: Dict[str, Dict[str,float]] = {}

        orig_label = f"{ops_orig.flute_data.flute_model} (Original)"
        analysis_list_for_plot.append((ops_orig.flute_data.acoustic_analysis, orig_label))
        if ops_orig.flute_data.finger_frequencies: # Comprobar que no es None
            finger_freqs_map_for_plot[orig_label] = ops_orig.flute_data.finger_frequencies

        if ops_mod and ops_mod.flute_data.acoustic_analysis: # Solo añadir si existe y tiene análisis
            mod_label = f"{ops_mod.flute_data.flute_model}" # FluteData ya le pone _mod
            if not mod_label.endswith(("(Modificada)")): # Para asegurar que se distinga en leyenda
                 mod_label += " (Modificada)"
            analysis_list_for_plot.append((ops_mod.flute_data.acoustic_analysis, mod_label))
            if ops_mod.flute_data.finger_frequencies:
                finger_freqs_map_for_plot[mod_label] = ops_mod.flute_data.finger_frequencies

        notes_ordered = []
        if ops_orig.flute_data.finger_frequencies: # Usar notas del original como referencia
            # Podríamos intentar un orden canónico
            canonical_order = ["D", "D#", "E", "F", "Fs", "G", "G#", "A", "A#", "B", "C", "Cs"]
            present_notes = ops_orig.flute_data.finger_frequencies.keys()
            notes_ordered = [n for n in canonical_order if n in present_notes]
            notes_ordered.extend(sorted(list(set(present_notes) - set(notes_ordered))))
        
        if not notes_ordered:
            logger.warning("No se pudieron determinar las notas ordenadas para los gráficos acústicos.")
            # Configurar placeholders para los gráficos acústicos si no hay notas
            self._configure_plot_axes_placeholders(self.ax_inharmonic)
            self._configure_plot_axes_placeholders(self.ax_moc)
            self._configure_plot_axes_placeholders(self.ax_bi_espe)
            return


        plot_configs = [
            (self.ax_inharmonic, FluteOperations.plot_summary_cents_differences, self.canvas_inharmonic, "Inharmonicity"),
            (self.ax_moc, FluteOperations.plot_moc_summary, self.canvas_moc, "MOC"),
            (self.ax_bi_espe, FluteOperations.plot_bi_espe_summary, self.canvas_bi_espe, "B_I / ESPE")
        ]

        for ax, plot_func, canvas, title_prefix in plot_configs:
            # ax.clear() # No es necesario si el método de ploteo ya lo hace y acepta 'ax'
            if notes_ordered and analysis_list_for_plot and finger_freqs_map_for_plot:
                try:
                    kwargs_for_plot = {
                        "acoustic_analysis_list": analysis_list_for_plot,
                        "notes_ordered": notes_ordered,
                        "ax": ax
                    }
                    # Añadir finger_frequencies_map solo si la función lo espera
                    if plot_func in [FluteOperations.plot_moc_summary, FluteOperations.plot_bi_espe_summary]:
                        if finger_freqs_map_for_plot:
                            kwargs_for_plot["finger_frequencies_map"] = finger_freqs_map_for_plot
                        else:
                            logger.warning(f"El mapa finger_frequencies está vacío o es None para {title_prefix}. Saltando este gráfico o podría fallar.")
                            self._configure_plot_axes_placeholders(specific_ax=ax)
                            canvas.draw_idle()
                            continue # Saltar este gráfico si el mapa esencial falta

                    plot_func(**kwargs_for_plot) # Llamar con argumentos desempaquetados

                    ax.set_title(f"{title_prefix}: {self.flute_name}", fontsize=10) # El título principal del eje
                except Exception as e:
                    logger.error(f"Error updating {title_prefix} plot for {self.flute_name}: {e}")
                    ax.clear()
                    ax.text(0.5,0.5, f"Error al graficar {title_prefix}", ha='center', va='center', color='red', transform=ax.transAxes)
                    ax.set_title(f"{title_prefix}: Error", fontsize=10)
            else:
                self._configure_plot_axes_placeholders(specific_ax=ax)
            canvas.draw_idle()


    def _save_modified_as(self):
        if not self.modified_flute_data_dict:
            messagebox.showerror("Error", "No modified data to save.", parent=self)
            return

        default_new_name = f"{self.flute_name}_mod" if self.flute_name else "modified_flute"
        new_name_suggestion = simpledialog.askstring(
            "Save Modified Geometry As",
            f"Enter name for the modified flute version (this will create a new directory):",
            initialvalue=default_new_name,
            parent=self
        )
        if not new_name_suggestion or not new_name_suggestion.strip():
            messagebox.showwarning("Save Cancelled", "No name provided for the new flute directory.", parent=self)
            return

        new_name_clean = new_name_suggestion.strip().replace(" ", "_") # Reemplazar espacios para nombre de dir
        
        # Determine base directory for saving: prefer parent of loaded data, else default data dir, else current dir
        base_save_dir_str = os.path.dirname(self.data_path) if self.data_path and os.path.isdir(os.path.dirname(self.data_path)) \
                            else (str(DEFAULT_DATA_DIR_EXPERIMENTER) if DEFAULT_DATA_DIR_EXPERIMENTER.exists() else ".")
                            
        if not base_save_dir_str: # Si data_path era solo un nombre de archivo
             base_save_dir_str = "." # Guardar en el directorio actual por defecto

        new_dir_path_obj = Path(base_save_dir_str) / new_name_clean

        if new_dir_path_obj.exists():
             if not messagebox.askyesno("Directory Exists",
                                        f"The directory '{new_dir_path_obj.name}' already exists in:\n'{new_dir_path_obj.parent}'.\n\n"
                                        "Overwrite its contents (JSON files for headjoint, left, right, foot)?",
                                        parent=self, icon=messagebox.WARNING):
                return
        try:
            new_dir_path_obj.mkdir(parents=True, exist_ok=True)

            from constants import FLUTE_PARTS_ORDER
            data_to_save = copy.deepcopy(self.modified_flute_data_dict) # Trabajar con una copia para ordenar

            for part_name in FLUTE_PARTS_ORDER:
                part_data = data_to_save.get(part_name)
                if isinstance(part_data, dict):
                    # Ordenar mediciones
                    if 'measurements' in part_data and isinstance(part_data['measurements'], list):
                        part_data['measurements'].sort(key=lambda item: item.get('position', 0.0))

                    # Ordenar agujeros si existen y son listas paralelas
                    if 'Holes position' in part_data and isinstance(part_data['Holes position'], list) and \
                       'Holes diameter' in part_data and isinstance(part_data['Holes diameter'], list):
                        pos_list = part_data['Holes position']; diam_list = part_data['Holes diameter']
                        # Asegurar que otras listas de agujeros (chimney, etc.) se ordenen igual si existen
                        # Esta parte puede ser compleja si hay muchas listas paralelas para los agujeros

                        if len(pos_list) == len(diam_list) and len(pos_list) > 0:
                            try:
                                # Crear lista de tuplas (pos, diam, otras_props...) para ordenar
                                hole_properties = ['Holes diameter'] # Lista base
                                # Añadir otras propiedades si existen y tienen la misma longitud
                                other_hole_props_names = ['Holes chimney', 'Holes diameter_out'] # etc.
                                temp_hole_data_list = []

                                for i in range(len(pos_list)):
                                    hole_entry = {'position': pos_list[i]}
                                    for prop_name in hole_properties:
                                        prop_list = part_data.get(prop_name, [])
                                        if i < len(prop_list): hole_entry[prop_name.split()[-1]] = prop_list[i] # 'diameter'
                                    # Para otras propiedades
                                    for other_prop_name in other_hole_props_names:
                                         other_prop_list = part_data.get(other_prop_name, [])
                                         if i < len(other_prop_list): hole_entry[other_prop_name.split()[-1]] = other_prop_list[i]


                                    temp_hole_data_list.append(hole_entry)

                                # Ordenar por posición
                                if all('position' in h_entry for h_entry in temp_hole_data_list):
                                    temp_hole_data_list.sort(key=lambda h_entry: float(h_entry['position']))

                                # Reasignar las listas ordenadas
                                part_data['Holes position'] = [h['position'] for h in temp_hole_data_list]
                                for prop_name in hole_properties:
                                     part_data[prop_name] = [h.get(prop_name.split()[-1]) for h in temp_hole_data_list]
                                for other_prop_name in other_hole_props_names:
                                     key_name = other_prop_name.split()[-1]
                                     if any(key_name in h_entry for h_entry in temp_hole_data_list): # Solo si la propiedad existía
                                         part_data[other_prop_name] = [h.get(key_name) for h in temp_hole_data_list]


                            except Exception as e_sort_save:
                                logger.warning(f"Could not fully sort all hole properties for {part_name} before saving: {e_sort_save}")


                    file_path = new_dir_path_obj / f"{part_name}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(part_data, f, indent=4, ensure_ascii=False)
                    logger.info(f"Saved {part_name}.json to {file_path}")
                else:
                    logger.warning(f"Part {part_name} not found in modified_flute_data_dict or is not a dict, cannot save for this part.")

            messagebox.showinfo("Save Successful", f"Modified geometry saved in directory:\n'{new_dir_path_obj}'", parent=self)
        except Exception as e:
            logger.exception(f"Error saving modified geometry to {new_dir_path_obj}")
            messagebox.showerror("Save Error", f"Could not save modified geometry: {e}", parent=self)

    def _on_close_app(self):
        quit_app = True
        if self.has_modifications:
            msg = "You have unsaved modifications."
            if self.data_modified_since_analysis:
                 msg += " These modifications have not been analyzed."
            msg += "\nQuit anyway?"

            if not messagebox.askyesno("Confirm Quit", msg, parent=self, icon=messagebox.WARNING):
                quit_app = False

        if quit_app:
            logger.info("Closing Flute Experimenter application.")
            plt.close('all')
            self.destroy()


if __name__ == "__main__":
    try:
        logger.info("Performing FluteData dictionary initialization test...")
        test_flute_dict = {
            "Flute Model": "TestDictFluteOnInit",
            "headjoint": {
                "Total length": 200.0, "Mortise length": 10.0,
                "measurements": [{"position": 0, "diameter": 20}, {"position": 190, "diameter": 18}],
                "Holes position": [], "Holes diameter": [], "Holes chimney": [], "Holes diameter_out": []
            },
            "left": {
                "Total length": 250.0, "Mortise length": 10.0,
                "measurements": [{"position": 0, "diameter": 18}, {"position": 240, "diameter": 18}],
                "Holes position": [], "Holes diameter": [], "Holes chimney": [], "Holes diameter_out": []
            },
            "right": {
                "Total length": 230.0, "Mortise length": 10.0,
                "measurements": [{"position": 0, "diameter": 18}, {"position": 220, "diameter": 18}],
                "Holes position": [], "Holes diameter": [], "Holes chimney": [], "Holes diameter_out": []
            },
            "foot": {
                "Total length": 100.0, "Mortise length": 10.0,
                "measurements": [{"position": 0, "diameter": 18}, {"position": 90, "diameter": 17}],
                "Holes position": [], "Holes diameter": [], "Holes chimney": [], "Holes diameter_out": []
            }
        }
        from flute_data import DEFAULT_FING_CHART_PATH # Importar la constante
        fing_chart_path_for_test = DEFAULT_FING_CHART_PATH 
        if not Path(fing_chart_path_for_test).exists():
            logger.warning(f"Default fingering chart '{fing_chart_path_for_test}' not found. Acoustic analysis might fail for FluteData test.")
            # Podrías crear un archivo de digitación dummy temporal aquí si es crítico para la prueba __init__
            # o asegurar que FluteData maneje elegantemente un fing_chart_file inexistente (ya lo hace con un error logueado)

        _ = FluteData(source=test_flute_dict, source_name="TestCheckFluteInMemory", fing_chart_file=fing_chart_path_for_test)
        logger.info("FluteData dictionary initialization test passed.")
    except Exception as e_test:
        logger.error(f"FATAL: FluteData dictionary initialization test failed: {e_test}", exc_info=True)
        messagebox.showerror("Startup Error",f"FluteData class failed dictionary initialization test:\n{type(e_test).__name__}: {e_test}\n\nCheck console logs for details. The application might not work correctly.")
        # Considerar no salir para permitir al usuario ver la UI, o salir:
        # exit()

    logger.info("Starting Flute Experimenter Application")
    app = FluteExperimentApp()
    app.mainloop()
    logger.info("Flute Experimenter Application closed")