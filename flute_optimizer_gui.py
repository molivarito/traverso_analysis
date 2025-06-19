#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flute_optimizer_gui.py

Interfaz gráfica para cargar una flauta desde JSON, optimizar la altura
de la chimenea de la embocadura y mostrar los resultados.
Usa datos de presión/flujo precalculados.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Optional, Dict, Any

# Intentar importar módulos principales primero
try:
    from optimize_flute_from_json import optimize_flute_from_json_full, plot_optimized_admittances
    from openwind import InstrumentPhysics, InstrumentGeometry, ImpedanceComputation # Importar ImpedanceComputation
    from constants import MM_TO_M_FACTOR as MM_TO_M_FACTOR_CONST, M_TO_MM_FACTOR as M_TO_MM_FACTOR_CONST
    from flute_data import FluteData, FluteDataInitializationError # <--- AÑADIDO
    from flute_operations import FluteOperations # Para plot_shape_static y plot_holes_static
except ImportError as e:
    # Este es un error crítico si estos módulos base no se encuentran.
    messagebox.showerror("Error de Importación Crítico",
                         f"No se pudieron importar módulos esenciales: {e}\n"
                         "Asegúrate de que los archivos del proyecto (optimize_flute_from_json.py, constants.py, etc.) "
                         "y la biblioteca OpenWind principal estén instalados y accesibles.")
    exit()

# Intentar importar tipos específicos de OpenWind para type checking y ploteo avanzado.
# Si fallan, el programa puede continuar con funcionalidad degradada.
try:
    from openwind.design.cone import Cone
except ImportError:
    logging.warning("No se pudo importar 'Cone' desde 'openwind.design'. El ploteo de conos 'raw' podría estar limitado.")
    Cone = None # Define Cone como None si la importación falla
try:
    from openwind.design.cylinder import Cylinder
except ImportError:
    logging.warning("No se pudo importar 'Cylinder' desde 'openwind.design'. El ploteo de cilindros 'raw' podría estar limitado.")
    Cylinder = None
try:
    from openwind.design.hole import Hole
except ImportError:
    logging.warning("No se pudo importar 'Hole' desde 'openwind.design'. El ploteo de agujeros 'raw' podría estar limitado.")
    Hole = None # Define Hole como None si la importación falla
try:
    from openwind.technical.instrument_geometry import SideHole, HoleCharacteristics
except ImportError:
    logging.warning("No se pudieron importar 'SideHole' o 'HoleCharacteristics' desde 'openwind.technical.instrument_geometry'. "
                    "El ploteo detallado de agujeros 'raw' en la vista lateral OW podría estar limitado.")
    SideHole = None
    HoleCharacteristics = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_JSON_DIR = SCRIPT_DIR.parent / "data_json"
if not DEFAULT_DATA_JSON_DIR.exists():
    DEFAULT_DATA_JSON_DIR = SCRIPT_DIR / "data_json"


class TraditionalTextEditor(tk.Toplevel): # Definición de la clase que faltaba
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Editor JSON Tradicional")
        self.geometry("800x600")
        self.filename = None
        self.create_widgets()

    def create_widgets(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        # btn_open = ttk.Button(toolbar, text="Abrir", command=self.open_file) # No necesario si se carga externamente
        # btn_open.pack(side=tk.LEFT, padx=2, pady=2)
        btn_save = ttk.Button(toolbar, text="Guardar y Cerrar", command=self.save_and_close)
        btn_save.pack(side=tk.LEFT, padx=5, pady=2)
        btn_cancel = ttk.Button(toolbar, text="Cancelar", command=self.cancel_edit)
        btn_cancel.pack(side=tk.LEFT, padx=5, pady=2)
        # btn_save_as = ttk.Button(toolbar, text="Guardar Como", command=self.save_as) # No necesario
        # btn_save_as.pack(side=tk.LEFT, padx=2, pady=2)
        # btn_close_file = ttk.Button(toolbar, text="Cerrar Archivo", command=self.close_file) # No necesario
        # btn_close_file.pack(side=tk.LEFT, padx=2, pady=2)
        # btn_exit = ttk.Button(toolbar, text="Salir Editor", command=self.exit_editor) # No necesario
        # btn_exit.pack(side=tk.LEFT, padx=2, pady=2)

        self.text_area = tk.Text(self, wrap=tk.WORD, undo=True) # Renombrado de self.text a self.text_area
        self.text_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        vsb = ttk.Scrollbar(self, orient="vertical", command=self.text_area.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.text_area.xview)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.text_area.configure(xscrollcommand=hsb.set)

    def load_file_content(self, filepath: str): # Método para cargar contenido
        self.filename = filepath
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert(tk.END, content)
            self.title(f"Editando - {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Error Abriendo Archivo", f"No se pudo cargar el archivo para editar:\n{e}", parent=self)
            self.destroy()

    def save_and_close(self): # Guardar y cerrar
        if self.filename:
            try:
                with open(self.filename, "w", encoding="utf-8") as f:
                    f.write(self.text_area.get("1.0", tk.END).strip() + "\n")
                logger.info(f"Archivo guardado: {self.filename}")
            except Exception as e:
                messagebox.showerror("Error Guardando", f"No se pudo guardar el archivo:\n{e}", parent=self)
                return # No cerrar si falla el guardado
        self.destroy()

    def cancel_edit(self): # Cancelar y cerrar
        self.destroy()


class FluteOptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimizador de Embocadura de Flauta")
        self.geometry("900x750")

        self.flute_dir_path: Optional[Path] = None
        self.flute_name: str = "Ninguna"

        self.optimized_chimney_heights: Optional[dict] = None
        self.initial_admittance_data_per_note: Optional[dict] = None
        self.optimized_admittance_data_per_note: Optional[dict] = None
        # self.physics_states_per_note: Optional[Dict[str, InstrumentPhysics]] = None # Ya no se usa directamente para el análisis optimizado
        self.pressure_flow_data_per_note: Optional[Dict[str, Dict[str, Any]]] = None # Para datos precalculados
        self.initial_acoustic_analysis_data: Optional[Dict[str, Any]] = None # Para guardar el análisis inicial
        self.optimized_acoustic_analysis_data: Optional[Dict[str, ImpedanceComputation]] = None # Ahora almacena ImpedanceComputation optimizados
        self.optimized_notes_list: Optional[list] = None
        self.target_frequencies_map: Optional[dict] = None

        self.protocol("WM_DELETE_WINDOW", self._on_closing_app)
        self._create_widgets()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        exit_button_main = ttk.Button(config_frame, text="Salir de Aplicación", command=self._on_closing_app)
        exit_button_main.grid(row=0, column=3, padx=20, pady=5, sticky="e")
        config_frame.pack(fill=tk.X, pady=5)

        load_button = ttk.Button(config_frame, text="Cargar Directorio de Flauta...", command=self._load_flute_dialog)
        load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.flute_name_label = ttk.Label(config_frame, text=f"Flauta Cargada: {self.flute_name}")
        self.flute_name_label.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(config_frame, text="Diapasón A4 (Hz):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.diapason_freq_var = tk.StringVar(value="415.0")
        self.diapason_entry = ttk.Entry(config_frame, textvariable=self.diapason_freq_var, width=10)
        self.diapason_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(config_frame, text="Temperatura (°C):").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.temperature_var = tk.StringVar(value="25.0")
        self.temperature_entry = ttk.Entry(config_frame, textvariable=self.temperature_var, width=10)
        self.temperature_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.optimize_button = ttk.Button(main_frame, text="Optimizar Embocadura", command=self._run_optimization, state=tk.DISABLED)
        self.optimize_button.pack(pady=10)

        self.results_notebook = ttk.Notebook(main_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        summary_text_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(summary_text_frame, text="Resumen (Texto)")
        self.results_text = tk.Text(summary_text_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_scrollbar = ttk.Scrollbar(summary_text_frame, orient="vertical", command=self.results_text.yview)
        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=summary_scrollbar.set)
        self.results_text.insert(tk.END, "Cargue una flauta y configure los parámetros para optimizar...\n")
        self.results_text.config(state=tk.DISABLED)

        self.chimney_plot_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.chimney_plot_frame, text="Alturas de Chimenea (Gráfico)")
        self.chimney_canvas_agg: Optional[FigureCanvasTkAgg] = None

        # Pestaña para Admitancia
        self.admittance_details_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.admittance_details_frame, text="Admitancia (por Nota)")

        admittance_controls_frame = ttk.Frame(self.admittance_details_frame, padding=(5,5))
        admittance_controls_frame.pack(fill=tk.X)
        ttk.Label(admittance_controls_frame, text="Seleccionar Nota:").pack(side=tk.LEFT, padx=(0,5))
        self.detailed_note_var = tk.StringVar()
        self.admittance_note_combobox = ttk.Combobox(admittance_controls_frame, textvariable=self.detailed_note_var, state="readonly", width=10)
        self.admittance_note_combobox.pack(side=tk.LEFT)
        self.admittance_note_combobox.bind("<<ComboboxSelected>>", self._update_detailed_plots_for_selected_note)

        self.admittance_plot_canvas_frame = ttk.Frame(self.admittance_details_frame, relief="sunken", borderwidth=1)
        self.admittance_plot_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.admittance_canvas_agg: Optional[FigureCanvasTkAgg] = None

        # Pestaña para Geometría y Modos P/F
        self.geometry_modes_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.geometry_modes_frame, text="Geometría y Modos (por Nota)")

        geom_modes_controls_frame = ttk.Frame(self.geometry_modes_frame, padding=(5,5))
        geom_modes_controls_frame.pack(fill=tk.X)
        ttk.Label(geom_modes_controls_frame, text="Seleccionar Nota:").pack(side=tk.LEFT, padx=(0,5))
        self.geom_modes_note_combobox = ttk.Combobox(geom_modes_controls_frame, textvariable=self.detailed_note_var, state="readonly", width=10)
        self.geom_modes_note_combobox.pack(side=tk.LEFT)
        self.geom_modes_note_combobox.bind("<<ComboboxSelected>>", self._update_detailed_plots_for_selected_note)

        geom_modes_paned_window = ttk.PanedWindow(self.geometry_modes_frame, orient=tk.VERTICAL)
        geom_modes_paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.detailed_geometry_plot_frame = ttk.Frame(geom_modes_paned_window, relief="sunken", borderwidth=1)
        geom_modes_paned_window.add(self.detailed_geometry_plot_frame, weight=1)
        self.geom_canvas_agg: Optional[FigureCanvasTkAgg] = None

        self.detailed_pressure_flow_plot_frame = ttk.Frame(geom_modes_paned_window, relief="sunken", borderwidth=1)
        geom_modes_paned_window.add(self.detailed_pressure_flow_plot_frame, weight=1)
        self.pf_canvas_agg: Optional[FigureCanvasTkAgg] = None

        # Pestaña para Resumen de Admitancias OpenWind
        self.ow_admittance_summary_tab_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.ow_admittance_summary_tab_frame, text="Resumen Admitancias OW")
        self.ow_admittance_summary_plot_frame = ttk.Frame(self.ow_admittance_summary_tab_frame, relief="sunken", borderwidth=1)
        self.ow_admittance_summary_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ow_admittance_summary_canvas_agg: Optional[FigureCanvasTkAgg] = None

        # Pestaña para Comparación de Inharmonicidad
        self.inharmonicity_comparison_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.inharmonicity_comparison_frame, text="Inharmonicidad (Antes/Después)")
        self.inharmonicity_canvas_agg: Optional[FigureCanvasTkAgg] = None

        # Pestaña para Geometría Detallada OpenWind (con su propio selector de nota) - Mantenida si es necesaria
        self.ow_detailed_geometry_tab_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.ow_detailed_geometry_tab_frame, text="Geometría Detallada OW (Nota)")

        ow_detailed_geom_controls_frame = ttk.Frame(self.ow_detailed_geometry_tab_frame, padding=(5,5))
        ow_detailed_geom_controls_frame.pack(fill=tk.X)
        ttk.Label(ow_detailed_geom_controls_frame, text="Seleccionar Nota:").pack(side=tk.LEFT, padx=(0,5))
        self.ow_detailed_geometry_note_var = tk.StringVar()
        self.ow_detailed_geometry_note_combobox = ttk.Combobox(ow_detailed_geom_controls_frame,
                                                               textvariable=self.ow_detailed_geometry_note_var,
                                                               state="readonly", width=10)
        self.ow_detailed_geometry_note_combobox.pack(side=tk.LEFT)
        self.ow_detailed_geometry_note_combobox.bind("<<ComboboxSelected>>", self._update_ow_detailed_geometry_plot)

        self.ow_detailed_geometry_plot_frame = ttk.Frame(self.ow_detailed_geometry_tab_frame, relief="sunken", borderwidth=1)
        self.ow_detailed_geometry_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ow_detailed_geometry_canvas_agg: Optional[FigureCanvasTkAgg] = None

    def _load_flute_dialog(self):
        dir_path = filedialog.askdirectory(
            title="Seleccionar Directorio de Datos de Flauta",
            initialdir=str(DEFAULT_DATA_JSON_DIR) if DEFAULT_DATA_JSON_DIR.exists() else "."
        )
        if dir_path:
            self.flute_dir_path = Path(dir_path)
            self.flute_name = self.flute_dir_path.name
            self.flute_name_label.config(text=f"Flauta Cargada: {self.flute_name}")
            self.optimize_button.config(state=tk.NORMAL)
            logger.info(f"Directorio de flauta seleccionado: {self.flute_dir_path}")
        else:
            self.flute_dir_path = None
            self.flute_name = "Ninguna"
            self.flute_name_label.config(text=f"Flauta Cargada: {self.flute_name}")
            self.optimize_button.config(state=tk.DISABLED)
            self.optimized_chimney_heights = None
            self.initial_admittance_data_per_note = None
            self.optimized_admittance_data_per_note = None
            # self.physics_states_per_note = None # Comentado
            self.pressure_flow_data_per_note = None
            self.initial_acoustic_analysis_data = None
            self.optimized_acoustic_analysis_data = None
            self.optimized_notes_list = None
            self._clear_all_plot_canvases()

    def _run_optimization(self):
        if not self.flute_dir_path:
            messagebox.showerror("Error", "Por favor, cargue primero un directorio de flauta.")
            return
        try:
            diapason_val = float(self.diapason_freq_var.get())
            temp_val = float(self.temperature_var.get())
            if diapason_val <= 0: raise ValueError("Frecuencia del diapasón debe ser positiva.")
        except ValueError as e:
            messagebox.showerror("Entrada Inválida", f"Valor de configuración inválido: {e}"); return

        self.optimized_chimney_heights = None
        self.initial_admittance_data_per_note = None
        self.optimized_admittance_data_per_note = None
        # self.physics_states_per_note = None # Comentado
        self.pressure_flow_data_per_note = None
        self.initial_acoustic_analysis_data = None
        self.optimized_acoustic_analysis_data = None
        self.optimized_notes_list = None
        self._clear_all_plot_canvases()
        
        # --- Carga interactiva de FluteData ---
        flute_data_obj_for_optim: Optional[FluteData] = None
        data_path_for_optim = self.flute_dir_path # Ya es un Path
        flute_dir_name_for_optim = self.flute_name

        while True: # Bucle para reintentar después de editar
            try:
                logger.info(f"Optimizador: Intentando cargar FluteData desde: {data_path_for_optim}")
                current_flute_data_attempt = FluteData(
                    str(data_path_for_optim),
                    source_name=flute_dir_name_for_optim,
                    la_frequency=diapason_val,
                    temperature=temp_val,
                    skip_acoustic_analysis=False 
                )

                if not current_flute_data_attempt.validation_errors:
                    if current_flute_data_attempt.validation_warnings:
                        warning_messages = "\n".join([w.get('message', 'Advertencia desconocida.') for w in current_flute_data_attempt.validation_warnings])
                        messagebox.showwarning("Advertencias de Validación", f"Advertencias para '{flute_dir_name_for_optim}':\n{warning_messages}", parent=self)
                    flute_data_obj_for_optim = current_flute_data_attempt
                    break 

                error_info = current_flute_data_attempt.validation_errors[0]
                error_message = error_info.get('message', 'Error desconocido.')
                part_with_error = error_info.get('part')
                file_to_edit_path_obj: Optional[Path] = None
                if part_with_error:
                    file_to_edit_path_obj = data_path_for_optim / f"{part_with_error}.json"

                prompt_message = f"Error en datos JSON para '{flute_dir_name_for_optim}':\n- {error_message}\n\n"
                
                if file_to_edit_path_obj and file_to_edit_path_obj.exists():
                    prompt_message += f"¿Desea editar el archivo '{file_to_edit_path_obj.name}' para corregirlo?"
                    user_choice = messagebox.askyesnocancel("Error de Datos JSON", prompt_message, parent=self, icon=messagebox.ERROR)
                    if user_choice is True: 
                        editor = TraditionalTextEditor(self) 
                        editor.load_file_content(str(file_to_edit_path_obj)) 
                        self.wait_window(editor)
                        continue 
                    elif user_choice is False: 
                        flute_data_obj_for_optim = None; break 
                    else: 
                        self.results_text.config(state=tk.NORMAL); self.results_text.insert(tk.END, "Optimización cancelada por el usuario.\n"); self.results_text.config(state=tk.DISABLED); return
                else: 
                    messagebox.showerror("Error de Datos JSON", prompt_message, parent=self)
                    flute_data_obj_for_optim = None; break
            except (FluteDataInitializationError, Exception) as e_load_fd:
                messagebox.showerror("Error de Carga de Flauta", f"Error al cargar datos para '{flute_dir_name_for_optim}':\n{e_load_fd}\n\nNo se puede optimizar.", parent=self)
                flute_data_obj_for_optim = None; break
        
        if not flute_data_obj_for_optim:
            self.results_text.config(state=tk.NORMAL); self.results_text.insert(tk.END, f"No se pudo cargar la flauta '{flute_dir_name_for_optim}' para optimización debido a errores.\n"); self.results_text.config(state=tk.DISABLED); return
        # --- Fin de Carga interactiva de FluteData ---

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"Optimizando para flauta: {self.flute_name}\n")
        self.results_text.insert(tk.END, f"Diapasón A4 = {diapason_val} Hz, Temperatura = {temp_val} °C\n\n")
        self.update_idletasks()
        try:
            self.optimized_chimney_heights, self.initial_admittance_data_per_note, \
            self.optimized_admittance_data_per_note, model_name, self.target_frequencies_map, diapason_used, \
            self.optimized_acoustic_analysis_data, self.pressure_flow_data_per_note, self.optimized_notes_list, \
            self.initial_acoustic_analysis_data = \
                optimize_flute_from_json_full( 
                    flute_data_instance=flute_data_obj_for_optim, 
                    diapason_a4_hz_gui=diapason_val,
                    target_temp_c_gui=temp_val
                )
            logger.debug(f"CHECKPOINT GUI: Después de optimize_flute_from_json_full.")
            logger.debug(f"  self.initial_acoustic_analysis_data: {{k: type(v).__name__ for k, v in self.initial_acoustic_analysis_data.items() if self.initial_acoustic_analysis_data}}")
            logger.debug(f"  self.optimized_acoustic_analysis_data: {{k: type(v).__name__ for k, v in self.optimized_acoustic_analysis_data.items() if self.optimized_acoustic_analysis_data}}")
            logger.debug(f"  self.optimized_notes_list: {self.optimized_notes_list}")


            self.results_text.insert(tk.END, "--- Alturas de Chimenea Optimizadas (mm) ---\n")
            if self.optimized_chimney_heights:
                for note, height in self.optimized_chimney_heights.items():
                    status = f"{height:.3f} mm" if not isinstance(height, float) or not np.isnan(height) else "Error en optimización"
                    self.results_text.insert(tk.END, f"  {note}: {status}\n")
                self._plot_chimney_heights_summary()
            else:
                self.results_text.insert(tk.END, "No se obtuvieron resultados de optimización.\n")

            # El bloque que construía self.optimized_acoustic_analysis_data a partir de self.physics_states_per_note se elimina
            # porque ahora se recibe directamente de optimize_flute_from_json_full.
            self._update_inharmonicity_plot() 
            self.results_text.insert(tk.END, "\nOptimización completada.\n")

            if self.optimized_notes_list:
                self.admittance_note_combobox['values'] = self.optimized_notes_list
                if hasattr(self, 'geom_modes_note_combobox'):
                    self.geom_modes_note_combobox['values'] = self.optimized_notes_list
                if hasattr(self, 'ow_detailed_geometry_note_combobox'):
                    self.ow_detailed_geometry_note_combobox['values'] = self.optimized_notes_list

                if self.optimized_notes_list:
                    self.detailed_note_var.set(self.optimized_notes_list[0])
                    self._update_detailed_plots_for_selected_note(event=None) # Esto actualizará Admitancia y P/F
                    self.ow_detailed_geometry_note_var.set(self.optimized_notes_list[0])
                    self._update_ow_detailed_geometry_plot(event=None)

                if self.optimized_admittance_data_per_note and self.flute_name and self.target_frequencies_map:
                    self._plot_openwind_admittance_summary()
            else:
                self.admittance_note_combobox['values'] = []
                if hasattr(self, 'geom_modes_note_combobox'): self.geom_modes_note_combobox['values'] = []
                if hasattr(self, 'ow_detailed_geometry_note_combobox'): self.ow_detailed_geometry_note_combobox['values'] = []
                self.detailed_note_var.set("")
        except FileNotFoundError as e_fnf:
            logger.error(f"Error de archivo durante la optimización: {e_fnf}", exc_info=True)
            messagebox.showerror("Error de Archivo", f"No se encontró un archivo necesario: {e_fnf}")
            self.results_text.insert(tk.END, f"\nERROR DE ARCHIVO: {e_fnf}\n")
        except Exception as e_opt:
            logger.error(f"Error durante el proceso de optimización: {e_opt}", exc_info=True)
            messagebox.showerror("Error de Optimización", f"Ocurrió un error: {e_opt}")
            self.results_text.insert(tk.END, f"\nERROR DE OPTIMIZACIÓN: {e_opt}\n")
        finally:
            self.results_text.config(state=tk.DISABLED)

    def _clear_plot_canvas(self, frame: ttk.Frame, canvas_agg_attr_name: str):
        canvas_agg = getattr(self, canvas_agg_attr_name, None)
        if canvas_agg:
            canvas_agg.get_tk_widget().destroy()
            setattr(self, canvas_agg_attr_name, None)
        for widget in frame.winfo_children():
            widget.destroy()

    def _clear_all_plot_canvases(self):
        self._clear_plot_canvas(self.chimney_plot_frame, "chimney_canvas_agg")
        self._clear_plot_canvas(self.admittance_plot_canvas_frame, "admittance_canvas_agg")
        self._clear_plot_canvas(self.detailed_pressure_flow_plot_frame, "pf_canvas_agg")
        self._clear_plot_canvas(self.detailed_geometry_plot_frame, "geom_canvas_agg")
        self._clear_plot_canvas(self.ow_admittance_summary_plot_frame, "ow_admittance_summary_canvas_agg")
        self._clear_plot_canvas(self.inharmonicity_comparison_frame, "inharmonicity_canvas_agg") 
        self._clear_plot_canvas(self.ow_detailed_geometry_plot_frame, "ow_detailed_geometry_canvas_agg")
        self.admittance_note_combobox['values'] = []
        if hasattr(self, 'geom_modes_note_combobox'): self.geom_modes_note_combobox['values'] = []
        if hasattr(self, 'ow_detailed_geometry_note_combobox'): self.ow_detailed_geometry_note_combobox['values'] = []
        self.detailed_note_var.set("")

    def _plot_chimney_heights_summary(self):
        self._clear_plot_canvas(self.chimney_plot_frame, "chimney_canvas_agg")
        if not self.optimized_chimney_heights: return

        notes = list(self.optimized_chimney_heights.keys())
        heights = [self.optimized_chimney_heights.get(n, 0) for n in notes]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(notes, heights, color='skyblue')
        ax.set_ylabel("Altura de Chimenea Optimizada (mm)")
        ax.set_title("Resumen de Alturas de Chimenea")
        ax.grid(True, axis='y', linestyle=':')
        fig.tight_layout()

        self.chimney_canvas_agg = FigureCanvasTkAgg(fig, master=self.chimney_plot_frame)
        self.chimney_canvas_agg.draw()
        self.chimney_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _update_inharmonicity_plot(self):
        self._clear_plot_canvas(self.inharmonicity_comparison_frame, "inharmonicity_canvas_agg")
        if not self.initial_acoustic_analysis_data or not self.optimized_acoustic_analysis_data or \
           not self.optimized_notes_list or not self.flute_name: 
            logger.warning("Datos insuficientes para plotear la inharmonicidad.")
            error_label = ttk.Label(self.inharmonicity_comparison_frame, text="No hay datos de inharmonicidad para mostrar.")
            error_label.pack(padx=10, pady=10, anchor="center")
            return

        try:
            fig_inharm = FluteOperations.plot_single_flute_inharmonicity_comparison(
                self.initial_acoustic_analysis_data, 
                self.optimized_acoustic_analysis_data, 
                self.optimized_notes_list, 
                self.flute_name 
            )
            if fig_inharm:
                self.inharmonicity_canvas_agg = FigureCanvasTkAgg(fig_inharm, master=self.inharmonicity_comparison_frame)
                self.inharmonicity_canvas_agg.draw()
                self.inharmonicity_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            else:
                logger.error("plot_single_flute_inharmonicity_comparison no devolvió una figura válida.")
                error_label = ttk.Label(self.inharmonicity_comparison_frame, text="Error al generar el gráfico de inharmonicidad.")
                error_label.pack(padx=10, pady=10, anchor="center")
        except Exception as e_inharm_plot:
            logger.error(f"Error al plotear la inharmonicidad: {e_inharm_plot}", exc_info=True)
            error_label = ttk.Label(self.inharmonicity_comparison_frame, text=f"Error al plotear la inharmonicidad:\n{e_inharm_plot}")
            error_label.pack(padx=10, pady=10, anchor="center")

    def _update_detailed_plots_for_selected_note(self, event: Optional[tk.Event]):
        selected_note = self.detailed_note_var.get()
        if not selected_note or not self.optimized_acoustic_analysis_data or not self.optimized_admittance_data_per_note or \
           not self.target_frequencies_map or not self.pressure_flow_data_per_note:
            self._clear_plot_canvas(self.admittance_plot_canvas_frame, "admittance_canvas_agg")
            self._clear_plot_canvas(self.detailed_pressure_flow_plot_frame, "pf_canvas_agg") 
            self._clear_plot_canvas(self.detailed_geometry_plot_frame, "geom_canvas_agg") 
            return

        self._clear_plot_canvas(self.admittance_plot_canvas_frame, "admittance_canvas_agg")
        fig_adm, ax_adm = plt.subplots(figsize=(7, 4))
        plot_adm_success = False
        initial_adm_tuple = self.initial_admittance_data_per_note.get(selected_note) if self.initial_admittance_data_per_note else None
        if initial_adm_tuple and initial_adm_tuple[0].size > 0:
            freqs_init, adm_db_init = initial_adm_tuple
            ax_adm.plot(freqs_init, adm_db_init, label=f"Admitancia Inicial ({selected_note})", color='gray', linestyle=':')
        optimized_adm_tuple = self.optimized_admittance_data_per_note.get(selected_note)
        if optimized_adm_tuple and optimized_adm_tuple[0].size > 0:
            freqs_opt, adm_db_opt = optimized_adm_tuple
            ax_adm.plot(freqs_opt, adm_db_opt, label=f"Admitancia Optimizada ({selected_note})", color='blue')
            plot_adm_success = True
        if plot_adm_success:
            target_f = self.target_frequencies_map.get(selected_note)
            if target_f: ax_adm.axvline(target_f, color='r', linestyle='--', label=f"Frec. Obj: {target_f:.1f} Hz")
            ax_adm.set_title(f"Admitancia para Nota {selected_note}"); ax_adm.set_xlabel("Frecuencia (Hz)"); ax_adm.set_ylabel("Admitancia (dB)")
            ax_adm.legend(); ax_adm.grid(True, linestyle=':')
        else:
            ax_adm.text(0.5, 0.5, "No hay datos de admitancia", ha='center', va='center', transform=ax_adm.transAxes)
            ax_adm.set_title(f"Admitancia para Nota {selected_note}")
        fig_adm.tight_layout()
        self.admittance_canvas_agg = FigureCanvasTkAgg(fig_adm, master=self.admittance_plot_canvas_frame)
        self.admittance_canvas_agg.draw(); self.admittance_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        current_optimized_ic = self.optimized_acoustic_analysis_data.get(selected_note) if self.optimized_acoustic_analysis_data else None
        pf_data_for_note = self.pressure_flow_data_per_note.get(selected_note)
        self._clear_plot_canvas(self.detailed_pressure_flow_plot_frame, "pf_canvas_agg")
        fig_pf, axs_pf = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        try:
            if pf_data_for_note and 'x_coords' in pf_data_for_note and \
               'pressure_modes' in pf_data_for_note and 'flow_modes' in pf_data_for_note and \
               'frequencies' in pf_data_for_note:
                x_coords = pf_data_for_note['x_coords']; pressure_modes = pf_data_for_note['pressure_modes']
                flow_modes = pf_data_for_note['flow_modes']; plot_frequencies = pf_data_for_note['frequencies']
                if x_coords is not None and pressure_modes is not None and flow_modes is not None and plot_frequencies is not None and x_coords.size > 0:
                    pressure_modes_to_plot = pressure_modes.T if pressure_modes.shape[0] == len(plot_frequencies) and pressure_modes.ndim == 2 else pressure_modes
                    flow_modes_to_plot = flow_modes.T if flow_modes.shape[0] == len(plot_frequencies) and flow_modes.ndim == 2 else flow_modes
                    if pressure_modes.ndim == 2 and pressure_modes.shape[1] != len(plot_frequencies):
                         logger.warning(f"Dimensiones de P/F no coinciden con frecuencias para {selected_note}. P: {pressure_modes.shape}, F: {flow_modes.shape}, Freqs: {len(plot_frequencies)}")
                    for i in range(min(pressure_modes_to_plot.shape[1], len(plot_frequencies))):
                        axs_pf[0].plot(x_coords, np.real(pressure_modes_to_plot[:, i]), label=f"Modo Frec: {plot_frequencies[i]:.0f} Hz")
                        axs_pf[1].plot(x_coords, np.real(flow_modes_to_plot[:, i]), label=f"Modo Frec: {plot_frequencies[i]:.0f} Hz")
                else: axs_pf[0].text(0.5, 0.5, "Datos P/F incompletos", ha='center', va='center', transform=axs_pf[0].transAxes)
            else: axs_pf[0].text(0.5, 0.5, "No hay datos P/F", ha='center', va='center', transform=axs_pf[0].transAxes)
            axs_pf[0].set_title(f"Presión Acústica ({selected_note})", fontsize=9); axs_pf[0].set_ylabel("Presión (Pa)"); axs_pf[0].grid(True, linestyle=':'); axs_pf[0].legend(fontsize='small')
            axs_pf[1].set_title(f"Flujo Acústico ({selected_note})", fontsize=9); axs_pf[1].set_xlabel("Posición (m)"); axs_pf[1].set_ylabel("Flujo (m³/s)"); axs_pf[1].grid(True, linestyle=':'); axs_pf[1].legend(fontsize='small')
        except Exception as e_pf:
            logger.error(f"Error graficando presión/flujo para {selected_note}: {e_pf}", exc_info=True)
            axs_pf[0].text(0.5, 0.5, "Error al graficar P/F", ha='center', va='center', transform=axs_pf[0].transAxes)
        fig_pf.tight_layout()
        self.pf_canvas_agg = FigureCanvasTkAgg(fig_pf, master=self.detailed_pressure_flow_plot_frame)
        self.pf_canvas_agg.draw(); self.pf_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._clear_plot_canvas(self.detailed_geometry_plot_frame, "geom_canvas_agg")
        fig_geom_simple, ax_geom_simple = plt.subplots(figsize=(7, 2.5))
        plot_geom_simple_success = False
        try:
            if current_optimized_ic and hasattr(current_optimized_ic, 'get_instrument_geometry') and isinstance(current_optimized_ic.get_instrument_geometry(), InstrumentGeometry): # type: ignore
                instrument_geometry_to_plot: InstrumentGeometry = current_optimized_ic.get_instrument_geometry() # type: ignore
                if hasattr(instrument_geometry_to_plot, 'get_xr_main_bore'): # type: ignore
                    x_coords_m_bore, r_coords_m_bore = instrument_geometry_to_plot.get_xr_main_bore() # type: ignore
                    x_coords_m_bore, r_coords_m_bore = np.array(x_coords_m_bore), np.array(r_coords_m_bore)
                    if x_coords_m_bore.size > 0 and r_coords_m_bore.size > 0:
                        min_len = min(len(x_coords_m_bore), len(r_coords_m_bore))
                        FluteOperations._plot_shape_static((x_coords_m_bore[:min_len], r_coords_m_bore[:min_len]), ax_geom_simple, 1.0, color='black', linewidth=1)
                        plot_geom_simple_success = True
                holes_details_for_plot = []
                fingering = instrument_geometry_to_plot.fingering_chart.fingering_of(selected_note) # type: ignore
                for hole_obj in instrument_geometry_to_plot.holes: # type: ignore
                    pos_m = hole_obj.position.get_value(); rad_m_val = 0.003
                    if hasattr(hole_obj.shape, 'get_radius_at'): rad_m_val = hole_obj.shape.get_radius_at(0)
                    elif hasattr(hole_obj, 'shape_char') and hasattr(hole_obj.shape_char, 'hole_shape') and hasattr(hole_obj.shape_char.hole_shape, 'radius'):
                        rad_param = hole_obj.shape_char.hole_shape.radius
                        rad_m_val = rad_param.get_value() if hasattr(rad_param, 'get_value') else rad_param
                    is_open = fingering.is_side_comp_open(hole_obj.label)
                    holes_details_for_plot.append({'label': hole_obj.label, 'position_m': pos_m, 'radius_m': rad_m_val, 'is_open': is_open})
                if holes_details_for_plot:
                    FluteOperations._plot_holes_static(holes_details_for_plot, ax_geom_simple, 1.0, default_color='dimgray', linewidth=0.5)
                    plot_geom_simple_success = True
                if plot_geom_simple_success:
                    ax_geom_simple.set_xlabel("Posición (m)"); ax_geom_simple.set_ylabel("Radio (m)")
                    ax_geom_simple.set_title(f"Geometría Simplificada ({selected_note})", fontsize=9); ax_geom_simple.grid(True, linestyle=':', alpha=0.7)
                    if hasattr(axs_pf[1], 'get_xlim'):
                        try: xlim_pf_m = axs_pf[1].get_xlim(); ax_geom_simple.set_xlim(xlim_pf_m)
                        except: pass
                    else: ax_geom_simple.autoscale_view(tight=True, scalex=True, scaley=False)
                    ax_geom_simple.autoscale_view(tight=None, scalex=False, scaley=True)
                else: ax_geom_simple.text(0.5, 0.5, "Geometría no disponible", ha='center', va='center', transform=ax_geom_simple.transAxes)
            else: ax_geom_simple.text(0.5, 0.5, "Geometría no disponible", ha='center', va='center', transform=ax_geom_simple.transAxes)
        except Exception as e_geom_simple:
            logger.error(f"Error graficando geometría simplificada para {selected_note}: {e_geom_simple}", exc_info=True)
            ax_geom_simple.text(0.5, 0.5, f"Error al graficar geometría:\n{e_geom_simple}", ha='center', va='center', transform=ax_geom_simple.transAxes, color='red')
        if not plot_geom_simple_success: ax_geom_simple.set_title(f"Geometría Simplificada ({selected_note}) - Error", fontsize=9)
        fig_geom_simple.tight_layout(pad=0.5)
        self.geom_canvas_agg = FigureCanvasTkAgg(fig_geom_simple, master=self.detailed_geometry_plot_frame)
        self.geom_canvas_agg.draw(); self.geom_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _update_ow_detailed_geometry_plot(self, event: Optional[tk.Event]):
        selected_note = self.ow_detailed_geometry_note_var.get()
        if not selected_note or not self.optimized_acoustic_analysis_data: # Usar optimized_acoustic_analysis_data
            self._clear_plot_canvas(self.ow_detailed_geometry_plot_frame, "ow_detailed_geometry_canvas_agg")
            if hasattr(self, 'ow_detailed_geometry_plot_frame'):
                error_label = ttk.Label(self.ow_detailed_geometry_plot_frame, text="Seleccione una nota o no hay datos de geometría optimizada.")
                error_label.pack(padx=10, pady=10, anchor="center")
            return
        self._clear_plot_canvas(self.ow_detailed_geometry_plot_frame, "ow_detailed_geometry_canvas_agg")
        
        current_optimized_ic_for_ow_geom = self.optimized_acoustic_analysis_data.get(selected_note) if self.optimized_acoustic_analysis_data else None

        if current_optimized_ic_for_ow_geom and \
           hasattr(current_optimized_ic_for_ow_geom, 'get_instrument_geometry') and \
           isinstance(current_optimized_ic_for_ow_geom.get_instrument_geometry(), InstrumentGeometry): # type: ignore
            instrument_geometry_ow_plot: InstrumentGeometry = current_optimized_ic_for_ow_geom.get_instrument_geometry() # type: ignore
            try:
                fig_ow_detailed = plt.figure()
                instrument_geometry_ow_plot.plot_InstrumentGeometry(figure=fig_ow_detailed, note=selected_note)
                if fig_ow_detailed and isinstance(fig_ow_detailed, plt.Figure) and hasattr(self, 'ow_detailed_geometry_plot_frame'):
                    self.ow_detailed_geometry_canvas_agg = FigureCanvasTkAgg(fig_ow_detailed, master=self.ow_detailed_geometry_plot_frame)
                    self.ow_detailed_geometry_canvas_agg.draw()
                    self.ow_detailed_geometry_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                elif hasattr(self, 'ow_detailed_geometry_plot_frame'):
                    error_label = ttk.Label(self.ow_detailed_geometry_plot_frame, text="Error: Geometría OW no disponible.")
                    error_label.pack(padx=10, pady=10, anchor="center")
            except Exception as e_plot_ow_detailed:
                logger.error(f"Error al plotear geometría OpenWind detallada para {selected_note}: {e_plot_ow_detailed}", exc_info=True)
                if hasattr(self, 'ow_detailed_geometry_plot_frame'):
                    error_label = ttk.Label(self.ow_detailed_geometry_plot_frame, text=f"Error al graficar geometría OW detallada:\n{e_plot_ow_detailed}")
                    error_label.pack(padx=10, pady=10, anchor="center")
        elif hasattr(self, 'ow_detailed_geometry_plot_frame'):
            error_label = ttk.Label(self.ow_detailed_geometry_plot_frame, text="Geometría OW detallada no disponible.")
            error_label.pack(padx=10, pady=10, anchor="center")

    def _plot_openwind_admittance_summary(self):
        self._clear_plot_canvas(self.ow_admittance_summary_plot_frame, "ow_admittance_summary_canvas_agg")
        if not self.optimized_admittance_data_per_note or not self.flute_name or \
           not self.target_frequencies_map or self.diapason_freq_var.get() == "":
            error_label = ttk.Label(self.ow_admittance_summary_plot_frame, text="No hay datos de admitancia optimizada para mostrar.")
            error_label.pack(padx=10, pady=10, anchor="center"); return
        try: diapason_val = float(self.diapason_freq_var.get())
        except ValueError:
            error_label = ttk.Label(self.ow_admittance_summary_plot_frame, text="Valor de diapasón inválido.")
            error_label.pack(padx=10, pady=10, anchor="center"); return
        fig_ow_adm_summary = plot_optimized_admittances(self.optimized_admittance_data_per_note, self.flute_name, self.target_frequencies_map, diapason_val, return_fig=True)
        if fig_ow_adm_summary:
            self.ow_admittance_summary_canvas_agg = FigureCanvasTkAgg(fig_ow_adm_summary, master=self.ow_admittance_summary_plot_frame)
            self.ow_admittance_summary_canvas_agg.draw()
            self.ow_admittance_summary_canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            error_label = ttk.Label(self.ow_admittance_summary_plot_frame, text="No hay datos de admitancia optimizada para mostrar.")
            error_label.pack(padx=10, pady=10, anchor="center")

    def _on_closing_app(self):
        for widget in self.winfo_children():
            if isinstance(widget, tk.Toplevel): widget.destroy()
        if messagebox.askokcancel("Salir", "¿Está seguro de que desea salir de la aplicación?"):
            plt.close('all'); self.destroy()

if __name__ == "__main__":
    app = FluteOptimizerApp()
    app.mainloop()
