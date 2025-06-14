#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
added_tube_flute.py

Aplicación para optimizar la altura de la chimenea de la embocadura
para que una flauta suene a frecuencias temperadas específicas.
(Versión de prueba con geometría hardcodeada del ejemplo de OpenWind)
"""

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import os
import json
import copy
from pathlib import Path
import numpy as np

from openwind import InstrumentGeometry, InstrumentPhysics, Player
from openwind.inversion import InverseFrequentialResponse 

try:
    # FluteData y FLUTE_PARTS_ORDER no se usan activamente en esta versión de prueba,
    # pero se mantienen las importaciones por si se reactivan más tarde.
    from flute_data import FluteData 
    from constants import FLUTE_PARTS_ORDER, M_TO_MM_FACTOR
    MM_TO_M_FACTOR = 0.001 
except ImportError:
    messagebox.showerror("Error de Importación", "No se pudieron importar FluteData o constants. Asegúrate de que están en la misma carpeta.")
    exit()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_JSON_DIR = SCRIPT_DIR / "data_json"
if not DEFAULT_DATA_JSON_DIR.exists():
    parent_dir_data = SCRIPT_DIR.parent / "data_json"
    if parent_dir_data.exists():
        DEFAULT_DATA_JSON_DIR = parent_dir_data
    else:
        try:
            DEFAULT_DATA_JSON_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directorio de datos por defecto creado en: {DEFAULT_DATA_JSON_DIR}")
        except Exception as e:
            logger.error(f"No se pudo crear el directorio de datos por defecto: {e}")


class ChimneyOptimizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Optimizador de Altura de Chimenea (Prueba Ejemplo OpenWind)")
        self.geometry("800x600")

        self.flute_data_instance: Optional[FluteData] = None 
        self.flute_name: str = "ExampleFlute (Hardcoded)" 
        self._create_widgets()
        self.flute_name_label.config(text=f"Usando: {self.flute_name}")


    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        load_config_frame = ttk.LabelFrame(main_frame, text="Configuración", padding="10")
        load_config_frame.pack(fill=tk.X, pady=5)

        load_button = ttk.Button(load_config_frame, text="Cargar Flauta (Directorio)...", command=self._load_flute_dialog, state=tk.DISABLED)
        load_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.flute_name_label = ttk.Label(load_config_frame, text=f"Usando: {self.flute_name}")
        self.flute_name_label.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        ttk.Label(load_config_frame, text="Diapasón (ej. A4 = Hz):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.diapason_freq_var = tk.StringVar(value="415.0") 
        self.diapason_entry = ttk.Entry(load_config_frame, textvariable=self.diapason_freq_var, width=10)
        self.diapason_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Button(main_frame, text="Optimizar Altura de Chimenea (Ejemplo)", command=self._calculate_optimized_chimney_heights).pack(pady=10)

        results_frame = ttk.LabelFrame(main_frame, text="Resultados (Altura de chimenea optimizada en mm)", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, height=15, width=80)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=results_scrollbar.set)
        self.results_text.insert(tk.END, "Los resultados aparecerán aquí...\n")
        self.results_text.config(state=tk.DISABLED)

    def _load_flute_dialog(self):
        messagebox.showinfo("Info", "La carga de flautas está deshabilitada en esta versión de prueba.\nSe utiliza una geometría de ejemplo hardcodeada.", parent=self)
        return

    def _calculate_optimized_chimney_heights(self):
        try:
            diapason_a4_hz = float(self.diapason_freq_var.get())
            if diapason_a4_hz <= 0: raise ValueError("Frecuencia del diapasón debe ser positiva.")
        except ValueError as e:
            messagebox.showerror("Entrada Inválida", f"Frecuencia del diapasón inválida: {e}"); return

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"Optimizando para flauta de ejemplo: {self.flute_name}\n")
        self.results_text.insert(tk.END, f"Diapasón A4 = {diapason_a4_hz} Hz\n\n")
        self.update_idletasks()

        bore_example_mm_diam = [[0,75, 10,10,'linear'], [75, 400, 10, 6,'linear']]
        holes_example_mm_diam = [['label','location','diameter','chimney'],
                                 ['embouchure',20,8,'1<~5'], 
                                 ['hole1',100,7,3], ['hole2',200,5,3], ['hole3',250,5,3],
                                 ['hole4',300,4,3], ['hole5',350,4,3]]
        chart_example = [['label', 'A', 'B', 'C', 'D'],
                         # La embocadura NO debe estar en el chart si es source_location
                         ['hole1','x','x','x','x'], ['hole2','x','x','x','o'],
                         ['hole3','x','x','o','o'], ['hole4','x','o','o','o'],
                         ['hole5','x','o','o','o']]

        my_geom = InstrumentGeometry(bore_example_mm_diam, holes_example_mm_diam, chart_example, 
                                     unit='mm', diameter=True)
        
        optim_params_obj = my_geom.optim_params
        if not optim_params_obj or not optim_params_obj.labels:
            messagebox.showerror("Error de Optimización", "No se encontraron parámetros optimizables en la geometría de ejemplo."); return
        logger.info(f"Parámetros optimizables detectados en my_geom (ejemplo): {optim_params_obj.labels}")

        notes_to_calculate = my_geom.fingering_chart.all_notes()
        semitone_map_example_relative_to_A = {'A': 0, 'B': 2, 'C': 3, 'D': 5}
        logger.info(f"Notas a calcular para el ejemplo: {notes_to_calculate}")

        embouchure_label = "embouchure"
        player_for_phy = Player() 

        physics_options = {
            'temperature': 25, 
            'player': player_for_phy, 
            'source_location': embouchure_label,
            'losses': True, 'humidity': 0.5,
            'radiation_category': {'entrance':'closed', 'holes':'unflanged', 'bell':'unflanged'}
        }
        my_phy = InstrumentPhysics(instrument_geometry=my_geom, **physics_options)
        
        # --- Checkpoint: Verificar y limpiar my_phy.optim_params ---
        # InstrumentPhysics hereda de OptimParams, y my_phy.optim_params es el objeto OptimParams fusionado.
        if hasattr(my_phy, 'optim_params') and my_phy.optim_params is not None:
            logger.info(f"Parámetros en my_phy.optim_params.labels ANTES de limpiar: {my_phy.optim_params.labels}")
            if 'rad_input' in my_phy.optim_params.labels:
                try:
                    my_phy.optim_params.delete_param('rad_input') 
                    logger.info("Se intentó eliminar 'rad_input' de my_phy.optim_params.labels.")
                    logger.info(f"Parámetros en my_phy.optim_params.labels DESPUÉS de limpiar: {my_phy.optim_params.labels}")
                except Exception as e_del_phy_param:
                    logger.warning(f"No se pudo eliminar 'rad_input' de my_phy.optim_params.labels: {e_del_phy_param}")
            else:
                logger.info("'rad_input' no encontrado en my_phy.optim_params.labels.")
        else:
            logger.warning("my_phy no tiene el atributo 'optim_params' o es None.")
                
        Z_target_resonance = np.array([0]) 
        initial_freq_for_optim_tool = diapason_a4_hz 
        optim_tool = InverseFrequentialResponse(my_phy, 
                                                initial_freq_for_optim_tool, 
                                                [Z_target_resonance], 
                                                observable='reflection')

        self.results_text.insert(tk.END, "Alturas de chimenea de embocadura optimizadas (mm):\n")
        initial_chimney_height_m_example = 0.001 # 1mm en metros

        for note_name in notes_to_calculate:
            if note_name not in semitone_map_example_relative_to_A:
                logger.warning(f"Nota '{note_name}' del chart de ejemplo no en semitone_map_example_relative_to_A. Saltando."); continue
            
            semitone_offset_from_A = semitone_map_example_relative_to_A[note_name]
            target_freq_hz = diapason_a4_hz * (2**(semitone_offset_from_A / 12.0))
            
            self.results_text.insert(tk.END, f"Nota {note_name}: Frec. Objetivo = {target_freq_hz:.2f} Hz. Optimizando...\n")
            self.update_idletasks()

            try:
                # El valor inicial debe establecerse en el objeto OptimParams que InverseFrequentialResponse usa,
                # que es my_phy.optim_params.
                # my_geom.optim_params (nuestro optim_params_obj) solo contiene los params de la geometría.
                if hasattr(my_phy, 'optim_params') and my_phy.optim_params is not None:
                    # Asegurarse de que el parámetro 'embouchure_chimney' exista antes de intentar establecerlo.
                    if 'embouchure_chimney' in my_phy.optim_params.labels:
                        my_phy.optim_params.set_active_values([initial_chimney_height_m_example], labels=['embouchure_chimney'])
                logger.debug(f"Reseteando altura de chimenea a {initial_chimney_height_m_example*M_TO_MM_FACTOR:.3f}mm para nota {note_name}")

                optim_tool.set_note(note_name)
                optim_tool.update_frequencies_and_mesh(np.array([target_freq_hz]))
                optim_tool.set_targets_list([Z_target_resonance], [note_name])
                optim_tool.optimize_freq_model(iter_detailed=True)
                
                optimized_chimney_height_m = optim_params_obj.get_active_values()[0]
                optimized_chimney_height_mm = optimized_chimney_height_m * M_TO_MM_FACTOR

                self.results_text.insert(tk.END, f"  Altura chimenea óptima = {optimized_chimney_height_mm:.3f} mm\n")

            except Exception as e_opt:
                logger.error(f"Error durante la optimización para nota {note_name}: {e_opt}", exc_info=True)
                self.results_text.insert(tk.END, f"  Error durante la optimización: {e_opt}\n")
            self.update_idletasks()

        self.results_text.insert(tk.END, "\nOptimización de ejemplo completada.\n")
        self.results_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = ChimneyOptimizerApp()
    app.mainloop()
