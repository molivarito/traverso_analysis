import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib
matplotlib.use('TkAgg') # Ensure Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json # Not strictly needed here, but good for context if data was from JSON
import copy
from typing import Dict, Any, Optional, Callable, List, Tuple

# Assuming these are available in the same directory or PYTHONPATH
from flute_data import FluteData # Used for type hinting if needed, not for instantiation here
from flute_operations import FluteOperations # Used for type hinting if needed
from constants import (
    FLUTE_PARTS_ORDER, M_TO_MM_FACTOR,
    DEFAULT_CHIMNEY_HEIGHT, DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT,
    DEFAULT_HOLE_RADIUS_OUT_FACTOR
)


import logging
logger = logging.getLogger(__name__)

class GraphicalFluteEditor(tk.Toplevel):
    def __init__(self, master: tk.Tk, initial_data: Dict[str, Any], flute_name: str, apply_callback: Callable[[Dict[str, Any]], None]):
        super().__init__(master)
        self.title(f"Edit Geometry: {flute_name}")
        self.geometry("1000x800") # Aumentar altura a 800

        self.initial_data = copy.deepcopy(initial_data)
        self.current_data = copy.deepcopy(initial_data)
        self.flute_name = flute_name
        self.apply_callback = apply_callback

        self._hole_add_y_threshold_mm = 5.0 # Umbral para añadir agujeros por clic
        self._y_offset_for_hole_markers_mm = 5.0 # Cuánto por debajo del perfil se dibujan los agujeros
        self._min_hole_marker_y_reference = 0 # Referencia Y para la línea de agujeros

        self._selected_part_name: str = FLUTE_PARTS_ORDER[0] if FLUTE_PARTS_ORDER else ""

        self.hole_artist_info: List[Dict[str, Any]] = []
        self.picked_hole_info: Optional[Dict[str, Any]] = None
        self._editor_combined_measurements: List[Dict[str, float]] = []

        self.bore_profile_point_artists_info: List[Dict[str, Any]] = []
        self.picked_bore_profile_point_info: Optional[Dict[str, Any]] = None
        self._drag_active = False
        self._is_dirty = False

        self._create_widgets()
        self._setup_plot()
        self._populate_editor_ui()
        self._update_plot()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self):
        control_panel = ttk.Frame(self, padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y)
        plot_panel = ttk.Frame(self, padding="10")
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.plot_panel = plot_panel

        control_notebook = ttk.Notebook(control_panel)
        control_notebook.pack(fill=tk.BOTH, expand=True)
        self.control_notebook = control_notebook
        self.part_frames: Dict[str, ttk.Frame] = {}
        self.part_measurement_entries: Dict[str, List[Tuple[ttk.Entry, ttk.Entry]]] = {}
        self.part_hole_entries: Dict[str, List[Tuple[ttk.Entry, ttk.Entry, ttk.Entry, ttk.Entry]]] = {}
        self.part_measurement_canvas_item_ids: Dict[str, int] = {}
        self.part_hole_canvas_item_ids: Dict[str, int] = {}
        self.part_length_entries: Dict[str, Dict[str, ttk.Entry]] = {}

        for part_name in FLUTE_PARTS_ORDER:
            part_frame = ttk.Frame(control_notebook, padding="5")
            control_notebook.add(part_frame, text=part_name.capitalize())
            self.part_frames[part_name] = part_frame
            self._build_part_editor_ui(part_frame, part_name)

        self.control_notebook.bind("<<NotebookTabChanged>>", self._on_part_tab_changed)

        button_frame = ttk.Frame(control_panel, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10,0))
        self.apply_button = ttk.Button(button_frame, text="Apply Changes", command=self._on_apply)
        self.apply_button.pack(side=tk.LEFT, padx=5)
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_close)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(button_frame, text="", foreground="blue")
        self.status_label.pack(side=tk.RIGHT, padx=5)

    def _build_part_editor_ui(self, parent_frame: ttk.Frame, part_name: str):
        length_frame = ttk.LabelFrame(parent_frame, text="Lengths (mm)", padding="5")
        length_frame.pack(fill=tk.X, pady=(0, 10))
        self.part_length_entries[part_name] = {}
        ttk.Label(length_frame, text="Total Length:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        total_len_entry = ttk.Entry(length_frame, width=10)
        total_len_entry.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        self.part_length_entries[part_name]['Total length'] = total_len_entry
        ttk.Label(length_frame, text="Mortise Length:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        mortise_len_entry = ttk.Entry(length_frame, width=10)
        mortise_len_entry.grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        self.part_length_entries[part_name]['Mortise length'] = mortise_len_entry
        length_frame.columnconfigure(1, weight=1)

        measurements_frame = ttk.LabelFrame(parent_frame, text="Measurements (mm)", padding="5")
        measurements_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        measurements_canvas = tk.Canvas(measurements_frame, borderwidth=0)
        measurements_scrollbar = ttk.Scrollbar(measurements_frame, orient="vertical", command=measurements_canvas.yview)
        measurements_scrollable_frame = ttk.Frame(measurements_canvas)
        measurements_canvas.configure(yscrollcommand=measurements_scrollbar.set)
        measurements_scrollbar.pack(side="right", fill="y")
        measurements_canvas.pack(side="left", fill="both", expand=True)
        self.part_measurement_canvas_item_ids[part_name] = measurements_canvas.create_window((0, 0), window=measurements_scrollable_frame, anchor="nw")
        measurements_scrollable_frame.bind("<Configure>", lambda e, c=measurements_canvas: c.configure(scrollregion = c.bbox("all")))
        measurements_canvas.bind("<Configure>", lambda e, c=measurements_canvas, item_id=self.part_measurement_canvas_item_ids[part_name]: c.itemconfig(item_id, width=e.width))
        self.part_measurement_entries[part_name] = []
        ttk.Label(measurements_scrollable_frame, text="Pos").grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(measurements_scrollable_frame, text="Diam").grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(measurements_scrollable_frame, text="").grid(row=0, column=2, padx=2, pady=2)
        meas_button_frame = ttk.Frame(measurements_frame)
        meas_button_frame.pack(fill=tk.X)
        ttk.Button(meas_button_frame, text="Add Measurement", command=lambda p=part_name, sf=measurements_scrollable_frame: self._add_measurement_entry(p, sf)).pack(side=tk.LEFT, padx=2)

        holes_frame = ttk.LabelFrame(parent_frame, text="Holes (mm)", padding="5")
        holes_frame.pack(fill=tk.BOTH, expand=True)
        holes_canvas = tk.Canvas(holes_frame, borderwidth=0)
        holes_scrollbar = ttk.Scrollbar(holes_frame, orient="vertical", command=holes_canvas.yview)
        holes_scrollable_frame = ttk.Frame(holes_canvas)
        holes_canvas.configure(yscrollcommand=holes_scrollbar.set)
        holes_scrollbar.pack(side="right", fill="y")
        holes_canvas.pack(side="left", fill="both", expand=True)
        self.part_hole_canvas_item_ids[part_name] = holes_canvas.create_window((0, 0), window=holes_scrollable_frame, anchor="nw")
        holes_scrollable_frame.bind("<Configure>", lambda e, c=holes_canvas: c.configure(scrollregion = c.bbox("all")))
        holes_canvas.bind("<Configure>", lambda e, c=holes_canvas, item_id=self.part_hole_canvas_item_ids[part_name]: c.itemconfig(item_id, width=e.width))
        self.part_hole_entries[part_name] = []
        ttk.Label(holes_scrollable_frame, text="Pos").grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="Diam").grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="Chimney").grid(row=0, column=2, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="Diam Out").grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="").grid(row=0, column=4, padx=2, pady=2)
        hole_button_frame = ttk.Frame(holes_frame)
        hole_button_frame.pack(fill=tk.X)
        ttk.Button(hole_button_frame, text="Add Hole", command=lambda p=part_name, sf=holes_scrollable_frame: self._add_hole_entry(p, sf)).pack(side=tk.LEFT, padx=2)

    def _add_measurement_entry(self, part_name: str, parent_frame: ttk.Frame, position: float = 0.0, diameter: float = 0.0):
        row_idx = len(self.part_measurement_entries[part_name]) + 1
        pos_entry = ttk.Entry(parent_frame, width=8)
        diam_entry = ttk.Entry(parent_frame, width=8)
        delete_button = ttk.Button(parent_frame, text="X", command=lambda p=part_name, r=row_idx-1, pf=parent_frame: self._remove_entry(p, 'measurement', r, pf), width=2)
        pos_entry.grid(row=row_idx, column=0, padx=2, pady=1); diam_entry.grid(row=row_idx, column=1, padx=2, pady=1); delete_button.grid(row=row_idx, column=2, padx=2, pady=1)
        pos_entry.insert(0, str(round(position,4))); diam_entry.insert(0, str(round(diameter,4)))
        self.part_measurement_entries[part_name].append((pos_entry, diam_entry))
        self._bind_modify_events([pos_entry, diam_entry])
        parent_frame.update_idletasks()
        if hasattr(parent_frame, 'master') and isinstance(parent_frame.master, tk.Canvas):
            parent_frame.master.configure(scrollregion=parent_frame.master.bbox("all"))

    def _add_hole_entry(self, part_name: str, parent_frame: ttk.Frame, position: float = 0.0, diameter: float = 0.0, chimney: float = 0.0, diameter_out: float = 0.0):
        row_idx = len(self.part_hole_entries[part_name]) + 1
        pos_entry = ttk.Entry(parent_frame, width=8); diam_entry = ttk.Entry(parent_frame, width=8)
        chimney_entry = ttk.Entry(parent_frame, width=8); diam_out_entry = ttk.Entry(parent_frame, width=8)
        delete_button = ttk.Button(parent_frame, text="X", command=lambda p=part_name, r=row_idx-1, pf=parent_frame: self._remove_entry(p, 'hole', r, pf), width=2)
        pos_entry.grid(row=row_idx, column=0, padx=2, pady=1); diam_entry.grid(row=row_idx, column=1, padx=2, pady=1)
        chimney_entry.grid(row=row_idx, column=2, padx=2, pady=1); diam_out_entry.grid(row=row_idx, column=3, padx=2, pady=1)
        delete_button.grid(row=row_idx, column=4, padx=2, pady=1)
        pos_entry.insert(0, str(round(position,4))); diam_entry.insert(0, str(round(diameter,4)))
        chimney_entry.insert(0, str(round(chimney,4))); diam_out_entry.insert(0, str(round(diameter_out,4)))
        self.part_hole_entries[part_name].append((pos_entry, diam_entry, chimney_entry, diam_out_entry))
        self._bind_modify_events([pos_entry, diam_entry, chimney_entry, diam_out_entry])
        parent_frame.update_idletasks()
        if hasattr(parent_frame, 'master') and isinstance(parent_frame.master, tk.Canvas):
            parent_frame.master.configure(scrollregion=parent_frame.master.bbox("all"))

    def _remove_entry(self, part_name: str, entry_type: str, index_to_remove: int, parent_frame: ttk.Frame):
        logger.debug(f"Attempting to remove {entry_type} at index {index_to_remove} from part {part_name}")
        entries_list_of_tuples = self.part_measurement_entries[part_name] if entry_type == 'measurement' else self.part_hole_entries[part_name]
        if not (0 <= index_to_remove < len(entries_list_of_tuples)):
            logger.warning(f"Invalid index {index_to_remove} for removing {entry_type}. List length: {len(entries_list_of_tuples)}")
            return
        for widget in parent_frame.grid_slaves(row=index_to_remove + 1): widget.destroy()
        entries_list_of_tuples.pop(index_to_remove)
        self._populate_editor_ui()
        self._set_dirty(True)
        self._collect_data_from_ui_and_update_current_data()
        self._update_plot()
        parent_frame.update_idletasks()
        if hasattr(parent_frame, 'master') and isinstance(parent_frame.master, tk.Canvas):
            parent_frame.master.configure(scrollregion=parent_frame.master.bbox("all"))

    def _populate_editor_ui(self):
        logger.debug(f"--- Iniciando _populate_editor_ui para la parte: {self._selected_part_name} ---")
        if not self.current_data: return
        part_name = self._selected_part_name
        part_data = self.current_data.get(part_name, {})

        if part_name in self.part_length_entries:
            for length_key, entry in self.part_length_entries[part_name].items():
                length_val = part_data.get(length_key, 0.0)
                entry.delete(0, tk.END); entry.insert(0, str(round(length_val,4)))
                self._bind_modify_events([entry])

        if part_name in self.part_measurement_entries:
            meas_scrollable_frame = self.part_frames[part_name].winfo_children()[1].winfo_children()[0].winfo_children()[0]
            for widget in meas_scrollable_frame.grid_slaves():
                if widget.grid_info()['row'] > 0: widget.destroy()
            self.part_measurement_entries[part_name] = []
            measurements = sorted(part_data.get("measurements", []), key=lambda item: item.get('position', 0.0))
            for meas in measurements:
                self._add_measurement_entry(part_name, meas_scrollable_frame, meas.get("position", 0.0), meas.get("diameter", 0.0))

        if part_name in self.part_hole_entries:
            holes_scrollable_frame = self.part_frames[part_name].winfo_children()[2].winfo_children()[0].winfo_children()[0]
            for widget in holes_scrollable_frame.grid_slaves():
                if widget.grid_info()['row'] > 0: widget.destroy()
            self.part_hole_entries[part_name] = []
            hole_data_list = []
            h_pos = part_data.get("Holes position", []); h_diam = part_data.get("Holes diameter", [])
            h_chim = part_data.get("Holes chimney", []); h_diam_o = part_data.get("Holes diameter_out", [])
            min_len = min(len(h_pos), len(h_diam))
            for i in range(min_len):
                hole_data_list.append({
                    'position': h_pos[i], 'diameter': h_diam[i],
                    'chimney': h_chim[i] if i < len(h_chim) else 0.0,
                    'diameter_out': h_diam_o[i] if i < len(h_diam_o) else 0.0
                })
            hole_data_list.sort(key=lambda item: item.get('position', 0.0))
            for hole in hole_data_list:
                self._add_hole_entry(part_name, holes_scrollable_frame, hole['position'], hole['diameter'], hole['chimney'], hole['diameter_out'])

    def _bind_modify_events(self, entries: List[ttk.Entry]):
        for entry in entries:
            entry.bind("<KeyRelease>", self._on_editor_modify)
            entry.bind("<FocusOut>", self._on_editor_modify)

    def _on_editor_modify(self, event=None):
        self._set_dirty(True)
        self._collect_data_from_ui_and_update_current_data()
        self._update_plot()

    def _set_dirty(self, is_dirty: bool):
        self._is_dirty = is_dirty
        self.status_label.config(text="Unsaved changes..." if is_dirty else "", foreground="red" if is_dirty else "blue")

    def _setup_plot(self):
        self.fig_plot, self.ax_plot = plt.subplots(figsize=(8, 6)) # Un solo Axes
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=self.plot_panel)
        self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
        self.canvas_plot_widget.pack(fill=tk.BOTH, expand=True)

        self.canvas_plot.mpl_connect('button_press_event', self._on_plot_button_press)
        self.canvas_plot.mpl_connect('pick_event', self._on_artist_pick)
        self.canvas_plot.mpl_connect('motion_notify_event', self._on_drag_motion)
        self.canvas_plot.mpl_connect('button_release_event', self._on_drag_release)
        self.fig_plot.tight_layout(pad=2.0)

    def _update_plot(self):
        logger.debug(f"--- Iniciando _update_plot para la parte: {self._selected_part_name} ---")
        if not self.current_data:
            self.ax_plot.clear()
            self.ax_plot.text(0.5, 0.5, "No data", ha='center', va='center', transform=self.ax_plot.transAxes)
            self.canvas_plot.draw_idle()
            return

        self.bore_profile_point_artists_info = []
        self.hole_artist_info = []
        part_name = self._selected_part_name
        part_data = self.current_data.get(part_name, {})

        self.ax_plot.clear()

        if not part_data:
            self.ax_plot.text(0.5, 0.5, f"No data for '{part_name}'", ha='center', va='center', transform=self.ax_plot.transAxes)
            self.ax_plot.set_title(f"Geometry: {part_name.capitalize()}", fontsize=10)
            self.canvas_plot.draw_idle()
            return

        # Plot Profile (Measurements)
        measurements = part_data.get("measurements", [])
        if isinstance(measurements, list):
            self._editor_combined_measurements = sorted(measurements, key=lambda item: item.get('position', 0.0))
        else:
            self._editor_combined_measurements = []

        min_profile_diameter = float('inf')
        if self._editor_combined_measurements:
            positions = [item["position"] for item in self._editor_combined_measurements]
            diameters = [item["diameter"] for item in self._editor_combined_measurements]
            if diameters: min_profile_diameter = min(diameters)

            self.ax_plot.plot(positions, diameters, label=f"Profile", color='blue', zorder=10)
            for idx, meas in enumerate(self._editor_combined_measurements):
                pos, diam = meas.get("position", 0.0), meas.get("diameter", 0.0)
                point_artist, = self.ax_plot.plot(pos, diam, 'o', color='skyblue', markersize=7, picker=5, alpha=0.7, zorder=11)
                self.bore_profile_point_artists_info.append({'artist': point_artist, 'part_name': part_name, 'measurement_index': idx})
        else: # No measurements
            min_profile_diameter = 10 # Default if no profile, for hole y_pos calculation

        # Plot Holes as markers below the profile
        hole_positions_rel_mm = part_data.get("Holes position", [])
        hole_diameters_mm = part_data.get("Holes diameter", [])
        
        # Determine Y position for hole markers
        self._min_hole_marker_y_reference = (min_profile_diameter if min_profile_diameter != float('inf') else 10) - self._y_offset_for_hole_markers_mm
        
        # Draw a faint line to indicate where holes are plotted
        if self._editor_combined_measurements: # Only draw if there's a profile
            xlims = self.ax_plot.get_xlim() if self.ax_plot.lines else (0, part_data.get("Total length", 100))
            self.ax_plot.hlines(self._min_hole_marker_y_reference, xlims[0], xlims[1], colors='grey', linestyles='dotted', alpha=0.5, zorder=1)


        if isinstance(hole_positions_rel_mm, list) and isinstance(hole_diameters_mm, list):
            for hole_idx_in_part, rel_pos_mm in enumerate(hole_positions_rel_mm):
                if hole_idx_in_part >= len(hole_diameters_mm): break
                hole_diam_mm = hole_diameters_mm[hole_idx_in_part]
                
                # Plot hole as a marker
                # Escalar el tamaño del marcador con el diámetro del agujero. Ajustar el factor 0.8 si es necesario.
                marker_size_scaled = max(hole_diam_mm * 0.8, 3) 
                hole_marker, = self.ax_plot.plot(rel_pos_mm, self._min_hole_marker_y_reference, 'o',
                                                 color='darkgreen', markersize=marker_size_scaled, picker=5, alpha=0.7, zorder=12)
                self.hole_artist_info.append({'artist': hole_marker, 'part_name': part_name, 'hole_index_in_part': hole_idx_in_part})

        self.ax_plot.set_title(f"Geometry: {part_name.capitalize()} - {self.flute_name}", fontsize=10)
        self.ax_plot.set_xlabel(f"Position in {part_name.capitalize()} (mm)")
        self.ax_plot.set_ylabel("Diameter / Hole Indication (mm)")
        self.ax_plot.grid(True, linestyle=':', alpha=0.7)
        if self.ax_plot.lines: self.ax_plot.legend(loc='best', fontsize=9)

        self.ax_plot.relim()
        self.ax_plot.autoscale_view(scalex=False, scaley=True)
        logger.debug(f"Plot Y-axis autoscaled. Limits: {self.ax_plot.get_ylim()}")

        self.canvas_plot.draw_idle()


    def _on_part_tab_changed(self, event=None):
        selected_tab_index = self.control_notebook.index("current")
        self._selected_part_name = FLUTE_PARTS_ORDER[selected_tab_index]
        logger.debug(f"Notebook tab changed to: {self._selected_part_name}")
        self._populate_editor_ui()
        self._update_plot()

    def _collect_data_from_ui_and_update_current_data(self):
        logger.debug(f"--- Iniciando _collect_data_from_ui_and_update_current_data para la parte: {self._selected_part_name} ---")
        part_name = self._selected_part_name; part_data_in_current = self.current_data.setdefault(part_name, {})
        if part_name in self.part_length_entries:
            for length_key, entry_widget in self.part_length_entries[part_name].items():
                try: part_data_in_current[length_key] = float(entry_widget.get())
                except ValueError: logger.warning(f"Invalid number for {length_key} in {part_name}.")
        if part_name in self.part_measurement_entries:
            measurements_list = []
            for i, (pos_entry, diam_entry) in enumerate(self.part_measurement_entries[part_name]):
                try: measurements_list.append({"position": float(pos_entry.get()), "diameter": float(diam_entry.get())})
                except ValueError: logger.warning(f"Invalid number for measurement {i+1} in {part_name}.")
            part_data_in_current['measurements'] = sorted(measurements_list, key=lambda item: item.get('position', 0.0))
        if part_name in self.part_hole_entries:
            hole_data_for_sort = []
            for i, (pos_e, diam_e, chim_e, diam_o_e) in enumerate(self.part_hole_entries[part_name]):
                try:
                    pos = float(pos_e.get()); diam = float(diam_e.get())
                    chim_s = chim_e.get(); diam_o_s = diam_o_e.get()
                    is_emb = (part_name == FLUTE_PARTS_ORDER[0] and i == 0)
                    def_chim = (DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT if is_emb else DEFAULT_CHIMNEY_HEIGHT) * M_TO_MM_FACTOR
                    chim = float(chim_s) if chim_s.strip() else def_chim
                    def_diam_o = diam * DEFAULT_HOLE_RADIUS_OUT_FACTOR if diam > 1e-6 else 0.1
                    diam_o = float(diam_o_s) if diam_o_s.strip() else def_diam_o
                    hole_data_for_sort.append({'position': pos, 'diameter': diam, 'chimney': chim, 'diameter_out': diam_o})
                except ValueError: logger.warning(f"Invalid number for hole {i+1} in {part_name}.")
            hole_data_for_sort.sort(key=lambda item: item.get('position', 0.0))
            part_data_in_current['Holes position'] = [h.get('position',0.0) for h in hole_data_for_sort]
            part_data_in_current['Holes diameter'] = [h.get('diameter',0.0) for h in hole_data_for_sort]
            part_data_in_current['Holes chimney'] = [h.get('chimney',0.0) for h in hole_data_for_sort]
            part_data_in_current['Holes diameter_out'] = [h.get('diameter_out',0.0) for h in hole_data_for_sort]
        logger.debug(f"self.current_data['{part_name}'] actualizado.")


    def _on_plot_button_press(self, event):
        # Si ya hay un arrastre activo con el botón izquierdo, no procesar nuevos clics izquierdos.
        # Los clics derechos para menú contextual deben permitirse.
        if event.button == 1 and self._drag_active:
            return
        if event.inaxes != self.ax_plot: return

        if event.button == 3:
            # El evento pick ya debería haber establecido picked_..._info si el clic derecho fue sobre un artista.
            # Si no, no hay nada seleccionado para mostrar menú.
            if self.picked_hole_info: self._show_hole_context_menu(event)
            elif self.picked_bore_profile_point_info: self._show_bore_point_context_menu(event)
            return

        if event.button == 1:
            if self.picked_bore_profile_point_info or self.picked_hole_info: # Un artista fue seleccionado por el evento pick
                self._drag_active = True
                logger.debug(f"Drag started on picked item.")
                return

            rel_x_mm, y_data_mm = event.xdata, event.ydata
            if rel_x_mm is None or y_data_mm is None: return

            part_name = self._selected_part_name
            part_data = self.current_data.get(part_name, {})
            part_total_length_mm = part_data.get("Total length", 0.0)
            clamped_relative_x_mm = max(0.0, min(rel_x_mm, part_total_length_mm))

            # Check if click is near the hole marker line
            if abs(y_data_mm - self._min_hole_marker_y_reference) < self._hole_add_y_threshold_mm :
                logger.info(f"Plot clicked near hole line to add hole at (rel_x={clamped_relative_x_mm:.2f}mm)")
                self._add_hole_from_plot_click(part_name, clamped_relative_x_mm)
            else: # Assume click is for adding a measurement point
                logger.info(f"Plot clicked to add measurement point at (rel_x={clamped_relative_x_mm:.2f}mm, diam={y_data_mm:.2f}mm)")
                new_measurement = {"position": round(clamped_relative_x_mm, 4), "diameter": round(y_data_mm, 4)}
                self.current_data[part_name].setdefault("measurements", []).append(new_measurement)
                self._set_dirty(True); self._populate_editor_ui(); self._update_plot()


    def _add_hole_from_plot_click(self, part_name: str, rel_x_mm: float):
        diameter_str = simpledialog.askstring("Add Hole", f"Enter diameter (mm) for new hole at {rel_x_mm:.2f}mm in {part_name}:", parent=self)
        if diameter_str is None: return
        try: diameter = float(diameter_str)
        except ValueError: messagebox.showwarning("Invalid Input", "Enter a valid number for diameter.", parent=self); return
        if diameter <= 0: messagebox.showwarning("Invalid Diameter", "Diameter must be positive.", parent=self); return

        part_data = self.current_data.setdefault(part_name, {})
        is_emb = (part_name == FLUTE_PARTS_ORDER[0] and len(part_data.get("Holes position", [])) == 0)
        def_chim = (DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT if is_emb else DEFAULT_CHIMNEY_HEIGHT) * M_TO_MM_FACTOR
        def_diam_o = diameter * DEFAULT_HOLE_RADIUS_OUT_FACTOR

        part_data.setdefault("Holes position", []).append(round(rel_x_mm, 4))
        part_data.setdefault("Holes diameter", []).append(round(diameter, 4))
        part_data.setdefault("Holes chimney", []).append(def_chim)
        part_data.setdefault("Holes diameter_out", []).append(def_diam_o)
        logger.info(f"Added hole at {rel_x_mm:.2f}mm, diam {diameter:.2f}mm to '{part_name}'")
        self._set_dirty(True); self._populate_editor_ui(); self._update_plot()

    def _show_hole_context_menu(self, event):
        if not self.picked_hole_info: return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Delete Hole", command=self._delete_picked_hole)
        # Usar event.guiEvent.x_root y event.guiEvent.y_root para obtener las coordenadas de pantalla
        # del evento de Matplotlib cuando se usa con TkAgg.
        try:
            if hasattr(event, 'guiEvent') and hasattr(event.guiEvent, 'x_root') and hasattr(event.guiEvent, 'y_root'):
                menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
            else: # Fallback o log de error si guiEvent no está disponible como se espera
                logger.warning("No se pudieron obtener las coordenadas x_root/y_root del evento para el menú contextual del agujero.")
        finally: menu.grab_release()

    def _delete_picked_hole(self):
        if not self.picked_hole_info: return
        part_name = self.picked_hole_info['part_name']; hole_idx = self.picked_hole_info['hole_index_in_part']
        part_data = self.current_data.get(part_name, {})
        for key in ["Holes position", "Holes diameter", "Holes chimney", "Holes diameter_out"]:
            if key in part_data and 0 <= hole_idx < len(part_data[key]): del part_data[key][hole_idx]
        logger.info(f"Deleted hole {hole_idx} from '{part_name}'."); self._set_dirty(True)
        self.picked_hole_info = None; self._populate_editor_ui(); self._update_plot()

    def _show_bore_point_context_menu(self, event):
        if not self.picked_bore_profile_point_info: return
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Delete Point", command=self._delete_picked_bore_point)
        try:
            if hasattr(event, 'guiEvent') and hasattr(event.guiEvent, 'x_root') and hasattr(event.guiEvent, 'y_root'):
                menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
            else:
                logger.warning("No se pudieron obtener las coordenadas x_root/y_root del evento para el menú contextual del punto de perfil.")
        finally: menu.grab_release()

    def _delete_picked_bore_point(self):
        if not self.picked_bore_profile_point_info: return
        part_name = self.picked_bore_profile_point_info['part_name']; meas_idx = self.picked_bore_profile_point_info['measurement_index']
        measurements = self.current_data.get(part_name, {}).get("measurements", [])
        if 0 <= meas_idx < len(measurements):
            del measurements[meas_idx]
            logger.info(f"Deleted measurement point {meas_idx} from '{part_name}'."); self._set_dirty(True)
        self.picked_bore_profile_point_info = None; self._populate_editor_ui(); self._update_plot()

    def _on_artist_pick(self, event):
        if self._drag_active: return
        artist = event.artist
        for point_info in self.bore_profile_point_artists_info:
            if point_info['artist'] == artist:
                # Seleccionar el punto, pero no activar _drag_active aquí.
                # _drag_active se activará en _on_plot_button_press si es un clic izquierdo.
                self.picked_bore_profile_point_info = point_info; self.picked_hole_info = None
                logger.debug(f"Picked bore point: {point_info}")
                return
        for hole_info in self.hole_artist_info:
            if hole_info['artist'] == artist:
                # Seleccionar el agujero, pero no activar _drag_active aquí.
                self.picked_hole_info = hole_info; self.picked_bore_profile_point_info = None
                logger.debug(f"Picked hole: {hole_info}")
                return

    def _on_drag_motion(self, event):
        if not self._drag_active or event.inaxes != self.ax_plot: return
        
        part_name = self._selected_part_name
        part_data = self.current_data.get(part_name, {})
        part_total_length_mm = part_data.get("Total length", 0.0)
        
        new_rel_x_mm = event.xdata
        if new_rel_x_mm is None: return
        clamped_relative_x_mm = max(0.0, min(new_rel_x_mm, part_total_length_mm))

        if self.picked_bore_profile_point_info:
            new_y_data = event.ydata
            if new_y_data is None: return

            artist_info = self.picked_bore_profile_point_info; artist = artist_info['artist']
            measurement_idx = artist_info['measurement_index']
            self.current_data[part_name]["measurements"][measurement_idx]["position"] = round(clamped_relative_x_mm, 4)
            self.current_data[part_name]["measurements"][measurement_idx]["diameter"] = round(new_y_data, 4)
            artist.set_data([clamped_relative_x_mm], [new_y_data])
            self.ax_plot.draw_artist(artist)
            self.canvas_plot.blit(self.ax_plot.bbox)

        elif self.picked_hole_info:
            artist_info = self.picked_hole_info; artist = artist_info['artist']
            hole_idx_in_part = artist_info['hole_index_in_part']
            self.current_data[part_name]["Holes position"][hole_idx_in_part] = round(clamped_relative_x_mm, 4)
            # Y position of hole markers is fixed, only X changes
            artist.set_data([clamped_relative_x_mm], [self._min_hole_marker_y_reference])
            self.ax_plot.draw_artist(artist)
            self.canvas_plot.blit(self.ax_plot.bbox)
        else:
            return
        
        self._set_dirty(True)

    def _on_drag_release(self, event):
        if self._drag_active:
            self._drag_active = False
            self.picked_bore_profile_point_info = None
            self.picked_hole_info = None
            self._populate_editor_ui()
            self._update_plot()
            logger.debug("Drag released.")

    def _on_apply(self):
        logger.info("Apply button clicked.")
        if not self._is_dirty:
            messagebox.showinfo("No Changes", "No changes to apply.", parent=self); return
        self._collect_data_from_ui_and_update_current_data()
        self._set_dirty(False)
        self.apply_callback(copy.deepcopy(self.current_data))

    def _on_close(self):
        if self._is_dirty and messagebox.askyesno("Unsaved Changes", "Discard unsaved changes and close editor?", parent=self, icon=messagebox.WARNING):
            self.destroy()
        elif not self._is_dirty:
            self.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    dummy_data = {
        "Flute Model": "TestFluteEditor",
        "headjoint": {"Total length": 200.0, "Mortise length": 10.0, "measurements": [{"position": 0.0, "diameter": 19.5}, {"position": 150.0, "diameter": 18.5}], "Holes position": [50.0], "Holes diameter": [10.0], "Holes chimney": [5.0], "Holes diameter_out": [12.0]},
        "left": {"Total length": 250.0, "Mortise length": 10.0, "measurements": [{"position": 0.0, "diameter": 18.5}, {"position": 240.0, "diameter": 17.5}], "Holes position": [30.0, 70.0], "Holes diameter": [8.0, 8.5], "Holes chimney": [3.0, 3.0], "Holes diameter_out": [9.0, 9.5]},
        "right": {"Total length": 200.0, "Mortise length": 10.0, "measurements": [], "Holes position": [], "Holes diameter": [], "Holes chimney": [], "Holes diameter_out": []},
        "foot": {"Total length": 100.0, "Mortise length": 10.0, "measurements": [], "Holes position": [], "Holes diameter": [], "Holes chimney": [], "Holes diameter_out": []}
    }
    def test_callback(updated_data: Dict[str, Any]): print("\nEditor Apply Callback - Updated data:", json.dumps(updated_data, indent=2))
    try:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        editor = GraphicalFluteEditor(root, dummy_data, "TestFluteEditor", test_callback)
        root.mainloop()
    except Exception as e: logger.exception("Error running editor test:"); print(f"Error: {e}")
