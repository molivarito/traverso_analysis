import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib
matplotlib.use('TkAgg') # Ensure Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import copy
from typing import Dict, Any, Optional, Callable, List, Tuple

# Assuming these are available in the same directory or PYTHONPATH
from flute_data import FluteData
from flute_operations import FluteOperations
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
        self.geometry("1000x700") # Adjust size as needed

        self.initial_data = copy.deepcopy(initial_data) # Keep a copy of the data passed in
        self.current_data = copy.deepcopy(initial_data) # This is what we will modify
        self.flute_name = flute_name
        self.apply_callback = apply_callback

        self._is_dirty = False # Track if changes have been made

        self._create_widgets()
        self._setup_plot()
        self._populate_editor_ui()
        self._update_plot() # Initial plot

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_widgets(self):
        # Main frames
        control_panel = ttk.Frame(self, padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y)

        plot_panel = ttk.Frame(self, padding="10")
        plot_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Control Panel - Simplified: Just Headjoint measurements for now
        control_notebook = ttk.Notebook(control_panel)
        control_notebook.pack(fill=tk.BOTH, expand=True)

        self.part_frames: Dict[str, ttk.Frame] = {}
        self.part_measurement_entries: Dict[str, List[Tuple[ttk.Entry, ttk.Entry]]] = {} # {part_name: [(pos_entry, diam_entry), ...]}
        self.part_hole_entries: Dict[str, List[Tuple[ttk.Entry, ttk.Entry, ttk.Entry, ttk.Entry]]] = {} # {part_name: [(pos_entry, diam_entry, chimney_entry, diam_out_entry), ...]}
        # Store canvas item IDs for embedded frames
        self.part_measurement_canvas_item_ids: Dict[str, int] = {}
        self.part_hole_canvas_item_ids: Dict[str, int] = {}
        self.part_length_entries: Dict[str, Dict[str, ttk.Entry]] = {} # {part_name: {'Total length': entry, 'Mortise length': entry}}

        for part_name in FLUTE_PARTS_ORDER:
            part_frame = ttk.Frame(control_notebook, padding="5")
            control_notebook.add(part_frame, text=part_name.capitalize())
            self.part_frames[part_name] = part_frame
            self._build_part_editor_ui(part_frame, part_name)

        # Buttons moved to control_panel
        button_frame = ttk.Frame(control_panel, padding="10")
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10,0))

        self.apply_button = ttk.Button(button_frame, text="Apply Changes", command=self._on_apply)
        self.apply_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self._on_close)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(button_frame, text="", foreground="blue")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        

    def _build_part_editor_ui(self, parent_frame: ttk.Frame, part_name: str):
        # Lengths
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


        # Measurements (Position, Diameter)
        measurements_frame = ttk.LabelFrame(parent_frame, text="Measurements (mm)", padding="5")
        measurements_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        measurements_canvas = tk.Canvas(measurements_frame, borderwidth=0)
        measurements_scrollbar = ttk.Scrollbar(measurements_frame, orient="vertical", command=measurements_canvas.yview)
        measurements_scrollable_frame = ttk.Frame(measurements_canvas)

        measurements_canvas.configure(yscrollcommand=measurements_scrollbar.set)

        measurements_scrollbar.pack(side="right", fill="y")
        measurements_canvas.pack(side="left", fill="both", expand=True)
        self.part_measurement_canvas_item_ids[part_name] = measurements_canvas.create_window((0, 0), window=measurements_scrollable_frame, anchor="nw")

        measurements_scrollable_frame.bind("<Configure>", lambda e: measurements_canvas.configure(scrollregion = measurements_canvas.bbox("all")))
        measurements_canvas.bind("<Configure>", lambda e, c=measurements_canvas, item_id=self.part_measurement_canvas_item_ids[part_name]: c.itemconfig(item_id, width=e.width))

        self.part_measurement_entries[part_name] = []
        ttk.Label(measurements_scrollable_frame, text="Pos").grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(measurements_scrollable_frame, text="Diam").grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(measurements_scrollable_frame, text="").grid(row=0, column=2, padx=2, pady=2) # For delete button

        # Add/Remove Measurement buttons
        meas_button_frame = ttk.Frame(measurements_frame)
        meas_button_frame.pack(fill=tk.X)
        ttk.Button(meas_button_frame, text="Add Measurement", command=lambda: self._add_measurement_entry(part_name, measurements_scrollable_frame)).pack(side=tk.LEFT, padx=2)


        # Holes (Position, Diameter, Chimney, Diameter_out)
        holes_frame = ttk.LabelFrame(parent_frame, text="Holes (mm)", padding="5")
        holes_frame.pack(fill=tk.BOTH, expand=True)

        holes_canvas = tk.Canvas(holes_frame, borderwidth=0)
        holes_scrollbar = ttk.Scrollbar(holes_frame, orient="vertical", command=holes_canvas.yview)
        holes_scrollable_frame = ttk.Frame(holes_canvas)

        holes_canvas.configure(yscrollcommand=holes_scrollbar.set)

        holes_scrollbar.pack(side="right", fill="y")
        holes_canvas.pack(side="left", fill="both", expand=True)
        self.part_hole_canvas_item_ids[part_name] = holes_canvas.create_window((0, 0), window=holes_scrollable_frame, anchor="nw")

        holes_scrollable_frame.bind("<Configure>", lambda e: holes_canvas.configure(scrollregion = holes_canvas.bbox("all")))
        holes_canvas.bind("<Configure>", lambda e, c=holes_canvas, item_id=self.part_hole_canvas_item_ids[part_name]: c.itemconfig(item_id, width=e.width))

        self.part_hole_entries[part_name] = []
        ttk.Label(holes_scrollable_frame, text="Pos").grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="Diam").grid(row=0, column=1, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="Chimney").grid(row=0, column=2, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="Diam Out").grid(row=0, column=3, padx=2, pady=2)
        ttk.Label(holes_scrollable_frame, text="").grid(row=0, column=4, padx=2, pady=2) # For delete button

        # Add/Remove Hole buttons
        hole_button_frame = ttk.Frame(holes_frame)
        hole_button_frame.pack(fill=tk.X)
        ttk.Button(hole_button_frame, text="Add Hole", command=lambda: self._add_hole_entry(part_name, holes_scrollable_frame)).pack(side=tk.LEFT, padx=2)


    def _add_measurement_entry(self, part_name: str, parent_frame: ttk.Frame, position: float = 0.0, diameter: float = 0.0):
        row_idx = len(self.part_measurement_entries[part_name]) + 1 # +1 because row 0 is headers
        pos_entry = ttk.Entry(parent_frame, width=8)
        diam_entry = ttk.Entry(parent_frame, width=8)
        delete_button = ttk.Button(parent_frame, text="X", command=lambda: self._remove_entry(part_name, 'measurement', row_idx - 1, parent_frame), width=2)

        pos_entry.grid(row=row_idx, column=0, padx=2, pady=1)
        diam_entry.grid(row=row_idx, column=1, padx=2, pady=1)
        delete_button.grid(row=row_idx, column=2, padx=2, pady=1)

        pos_entry.insert(0, str(position))
        diam_entry.insert(0, str(diameter))

        self.part_measurement_entries[part_name].append((pos_entry, diam_entry))
        self._bind_modify_events([pos_entry, diam_entry])
        parent_frame.update_idletasks() # Update layout before configuring scrollregion
        parent_frame.master.configure(scrollregion=parent_frame.master.bbox("all")) # Update canvas scrollregion

    def _add_hole_entry(self, part_name: str, parent_frame: ttk.Frame, position: float = 0.0, diameter: float = 0.0, chimney: float = 0.0, diameter_out: float = 0.0):
        row_idx = len(self.part_hole_entries[part_name]) + 1 # +1 because row 0 is headers
        pos_entry = ttk.Entry(parent_frame, width=8)
        diam_entry = ttk.Entry(parent_frame, width=8)
        chimney_entry = ttk.Entry(parent_frame, width=8)
        diam_out_entry = ttk.Entry(parent_frame, width=8)
        delete_button = ttk.Button(parent_frame, text="X", command=lambda: self._remove_entry(part_name, 'hole', row_idx - 1, parent_frame), width=2)

        pos_entry.grid(row=row_idx, column=0, padx=2, pady=1)
        diam_entry.grid(row=row_idx, column=1, padx=2, pady=1)
        chimney_entry.grid(row=row_idx, column=2, padx=2, pady=1)
        diam_out_entry.grid(row=row_idx, column=3, padx=2, pady=1)
        delete_button.grid(row=row_idx, column=4, padx=2, pady=1)

        pos_entry.insert(0, str(position))
        diam_entry.insert(0, str(diameter))
        chimney_entry.insert(0, str(chimney))
        diam_out_entry.insert(0, str(diameter_out))


        self.part_hole_entries[part_name].append((pos_entry, diam_entry, chimney_entry, diam_out_entry))
        self._bind_modify_events([pos_entry, diam_entry, chimney_entry, diam_out_entry])
        parent_frame.update_idletasks() # Update layout before configuring scrollregion
        parent_frame.master.configure(scrollregion=parent_frame.master.bbox("all")) # Update canvas scrollregion


    def _remove_entry(self, part_name: str, entry_type: str, index: int, parent_frame: ttk.Frame):
        if entry_type == 'measurement':
            entries_list = self.part_measurement_entries[part_name]
        elif entry_type == 'hole':
            entries_list = self.part_hole_entries[part_name]
        else:
            return

        if 0 <= index < len(entries_list):
            # Destroy the widgets in the row
            for entry in entries_list[index]:
                entry.destroy()
            # Destroy the delete button in the row (it's the last widget added for the row)
            # Find the delete button by checking widgets in the parent frame at the correct grid row
            widgets_in_row = parent_frame.grid_slaves(row=index + 1) # +1 because headers are row 0
            for widget in widgets_in_row:
                 if isinstance(widget, ttk.Button) and widget.cget('text') == 'X':
                      widget.destroy()
                      break # Assuming only one delete button per row

            # Remove the entry tuple from our list
            removed_entry = entries_list.pop(index)

            # Re-grid subsequent rows to fill the gap
            for i in range(index, len(entries_list)):
                row_widgets = parent_frame.grid_slaves(row=i + 2) # Widgets currently at row i+2
                for widget in row_widgets:
                    widget.grid(row=i + 1) # Move to row i+1

            self._set_dirty(True)
            self._update_plot() # Update plot after removing data
            parent_frame.update_idletasks() # Update layout before configuring scrollregion
            parent_frame.master.configure(scrollregion=parent_frame.master.bbox("all")) # Update canvas scrollregion


    def _populate_editor_ui(self):
        if not self.current_data: return

        for part_name in FLUTE_PARTS_ORDER:
            part_data = self.current_data.get(part_name, {})

            # Populate Lengths
            if part_name in self.part_length_entries:
                for length_key in ['Total length', 'Mortise length']:
                    entry = self.part_length_entries[part_name].get(length_key)
                    if entry:
                        length_val = part_data.get(length_key, 0.0)
                        entry.delete(0, tk.END)
                        entry.insert(0, str(length_val))
                        self._bind_modify_events([entry])


            # Populate Measurements
            if part_name in self.part_measurement_entries:
                # Clear existing entries first
                for entry_tuple in self.part_measurement_entries[part_name]:
                    for entry in entry_tuple: entry.destroy()
                # Also destroy delete buttons
                meas_frame = self.part_frames[part_name].winfo_children()[1] # Measurements frame
                meas_canvas = meas_frame.winfo_children()[0] # Canvas
                meas_scrollable_frame = meas_canvas.winfo_children()[0] # Scrollable frame
                for widget in meas_scrollable_frame.grid_slaves():
                    if isinstance(widget, ttk.Button) and widget.cget('text') == 'X':
                        widget.destroy()

                self.part_measurement_entries[part_name] = [] # Reset the list

                measurements = part_data.get("measurements", [])
                # Sort measurements by position before populating UI
                if isinstance(measurements, list):
                     try:
                         measurements.sort(key=lambda item: item.get('position', 0.0))
                     except Exception as e_sort_meas:
                         logger.warning(f"Could not sort measurements for part {part_name} during UI population: {e_sort_meas}")

                meas_scrollable_frame = self.part_frames[part_name].winfo_children()[1].winfo_children()[0].winfo_children()[0] # Path to scrollable frame
                for meas in measurements:
                    pos = meas.get("position", 0.0)
                    diam = meas.get("diameter", 0.0)
                    self._add_measurement_entry(part_name, meas_scrollable_frame, pos, diam)


            # Populate Holes
            if part_name in self.part_hole_entries:
                 # Clear existing entries first
                for entry_tuple in self.part_hole_entries[part_name]:
                    for entry in entry_tuple: entry.destroy()
                # Also destroy delete buttons
                holes_frame = self.part_frames[part_name].winfo_children()[2] # Holes frame
                holes_canvas = holes_frame.winfo_children()[0] # Canvas
                holes_scrollable_frame = holes_canvas.winfo_children()[0] # Scrollable frame
                for widget in holes_scrollable_frame.grid_slaves():
                    if isinstance(widget, ttk.Button) and widget.cget('text') == 'X':
                        widget.destroy()

                self.part_hole_entries[part_name] = [] # Reset the list

                hole_positions = part_data.get("Holes position", [])
                hole_diameters = part_data.get("Holes diameter", [])
                hole_chimneys = part_data.get("Holes chimney", [])
                hole_diameters_out = part_data.get("Holes diameter_out", [])

                # Combine hole data for sorting
                hole_data_list = []
                min_len = min(len(hole_positions), len(hole_diameters)) # Assume pos and diam are minimum required
                for i in range(min_len):
                    pos = hole_positions[i]
                    diam = hole_diameters[i]
                    chimney = hole_chimneys[i] if i < len(hole_chimneys) else 0.0 # Use 0.0 as default if missing
                    diam_out = hole_diameters_out[i] if i < len(hole_diameters_out) else 0.0 # Use 0.0 as default if missing
                    hole_data_list.append({'position': pos, 'diameter': diam, 'chimney': chimney, 'diameter_out': diam_out})

                # Sort holes by position
                try:
                    hole_data_list.sort(key=lambda item: item.get('position', 0.0))
                except Exception as e_sort_holes:
                    logger.warning(f"Could not sort holes for part {part_name} during UI population: {e_sort_holes}")


                holes_scrollable_frame = self.part_frames[part_name].winfo_children()[2].winfo_children()[0].winfo_children()[0] # Path to scrollable frame
                for hole in hole_data_list:
                    self._add_hole_entry(
                        part_name,
                        holes_scrollable_frame,
                        hole.get('position', 0.0),
                        hole.get('diameter', 0.0),
                        hole.get('chimney', 0.0),
                        hole.get('diameter_out', 0.0)
                    )

        self._set_dirty(False) # Reset dirty flag after populating
        self.status_label.config(text="")


    def _bind_modify_events(self, entries: List[ttk.Entry]):
        for entry in entries:
            entry.bind("<KeyRelease>", self._on_editor_modify)
            entry.bind("<FocusOut>", self._on_editor_modify) # Also trigger on losing focus

    def _on_editor_modify(self, event=None):
        self._set_dirty(True)
        self._update_plot() # Update plot preview on modification

    def _set_dirty(self, is_dirty: bool):
        self._is_dirty = is_dirty
        if is_dirty:
            self.status_label.config(text="Unsaved changes...", foreground="red")
        else:
            self.status_label.config(text="Saved.", foreground="green") # This label is for editor state, not main app state


    def _setup_plot(self):
        self.fig_plot, self.ax_plot = plt.subplots(figsize=(8, 5)) # Adjust size
        self.canvas_plot = FigureCanvasTkAgg(self.fig_plot, master=self.pack_slaves()[-1]) # Attach to the plot_panel
        self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
        self.canvas_plot_widget.pack(fill=tk.BOTH, expand=True)

    def _update_plot(self):
        logger.debug("Updating editor plot preview...")
        if not self.current_data:
            self.ax_plot.clear()
            self.ax_plot.text(0.5, 0.5, "No data to plot", ha='center', va='center', transform=self.ax_plot.transAxes)
            self.canvas_plot.draw_idle()
            return

        # Temporarily create a FluteData object from current_data for plotting
        try:
            # Need to ensure combined_measurements is updated from current_data
            # FluteData.__init__ calls combine_measurements, so just pass current_data
            # Need to pass a source_name to FluteData to avoid default "InMemoryFlute"
            temp_flute_data = FluteData(
                source=copy.deepcopy(self.current_data),
                source_name=f"{self.flute_name}_editor_preview",
                skip_acoustic_analysis=True) # <--- AÑADIDO: No calcular acústica para preview
            temp_flute_ops = FluteOperations(temp_flute_data)

            self.ax_plot.clear()
            # Use plot_combined_flute_data for the main profile view
            temp_flute_ops.plot_combined_flute_data(ax=self.ax_plot, plot_label="Current Geometry")

            # --- Visualizar Agujeros ---
            holes_details_for_plot = []
            current_part_start_abs_mm = 0.0
            hole_counter = 0 # For global hole numbering like hole1, hole2... (excluding embouchure)
            
            part_data_map = {p_name: self.current_data.get(p_name, {}) for p_name in FLUTE_PARTS_ORDER}

            for i, p_name in enumerate(FLUTE_PARTS_ORDER):
                part_specific_data = part_data_map.get(p_name)
                if not part_specific_data: continue

                if i > 0: # Calculate absolute start for parts after headjoint
                    headjoint_data = part_data_map.get(FLUTE_PARTS_ORDER[0], {})
                    hj_total_length = headjoint_data.get("Total length", 0.0)
                    hj_mortise = headjoint_data.get("Mortise length", 0.0)

                    left_data = part_data_map.get(FLUTE_PARTS_ORDER[1], {})
                    left_total_length = left_data.get("Total length", 0.0)
                    left_mortise = left_data.get("Mortise length", 0.0)

                    right_data = part_data_map.get(FLUTE_PARTS_ORDER[2], {})
                    right_total_length = right_data.get("Total length", 0.0)
                    right_mortise = right_data.get("Mortise length", 0.0)

                    if p_name == FLUTE_PARTS_ORDER[1]: # left
                        current_part_start_abs_mm = hj_total_length - hj_mortise
                    elif p_name == FLUTE_PARTS_ORDER[2]: # right
                        current_part_start_abs_mm = (hj_total_length - hj_mortise) + \
                                                  (left_total_length - left_mortise)
                    elif p_name == FLUTE_PARTS_ORDER[3]: # foot
                        current_part_start_abs_mm = (hj_total_length - hj_mortise) + \
                                                  (left_total_length - left_mortise) + \
                                                  (right_total_length - right_mortise)
                else: # headjoint
                    current_part_start_abs_mm = 0.0

                hole_positions_rel_mm = part_specific_data.get("Holes position", [])
                hole_diameters_mm = part_specific_data.get("Holes diameter", [])
                
                for j, rel_pos_mm in enumerate(hole_positions_rel_mm):
                    if j >= len(hole_diameters_mm): break
                    abs_pos_mm = current_part_start_abs_mm + rel_pos_mm
                    radius_mm = hole_diameters_mm[j] / 2.0
                    label = "embouchure" if p_name == FLUTE_PARTS_ORDER[0] and j == 0 else f"hole{hole_counter + 1}"
                    if not (p_name == FLUTE_PARTS_ORDER[0] and j == 0): hole_counter +=1
                    
                    holes_details_for_plot.append({
                        'label': label,
                        'position_m': abs_pos_mm / M_TO_MM_FACTOR,
                        'radius_m': radius_mm / M_TO_MM_FACTOR,
                        'is_open': True # Visual only, state doesn't matter here
                    })
            if holes_details_for_plot:
                FluteOperations._plot_holes_static(holes_details_for_plot, self.ax_plot, M_TO_MM_FACTOR, default_color='darkred', linewidth=0.5, alpha=0.6)
            # --- Fin Visualizar Agujeros ---
            
            self.ax_plot.set_title(f"Geometry Preview: {self.flute_name}", fontsize=10)
            self.ax_plot.set_xlabel("Position (mm)"); self.ax_plot.set_ylabel("Diameter (mm)")
            self.ax_plot.grid(True, linestyle=':', alpha=0.7)
            self.ax_plot.legend(loc='best', fontsize=9)

        except Exception as e:
            logger.error(f"Error updating editor plot: {e}")
            self.ax_plot.clear()
            self.ax_plot.text(0.5, 0.5, f"Plot Error: {type(e).__name__}", ha='center', va='center', color='red', transform=self.ax_plot.transAxes)
            self.ax_plot.set_title("Geometry Preview: Error", fontsize=10)

        self.canvas_plot.draw_idle()


    def _on_apply(self):
        logger.info("Apply button clicked.")
        if not self._is_dirty:
            messagebox.showinfo("No Changes", "No changes have been made to apply.", parent=self)
            self._on_close() # Close if no changes
            return

        # Collect data from UI and update self.current_data
        updated_data = copy.deepcopy(self.initial_data) # Start from initial to preserve structure/keys
        updated_data["Flute Model"] = self.flute_name # Ensure model name is consistent

        validation_errors = []

        for part_name in FLUTE_PARTS_ORDER:
            part_data = updated_data.get(part_name, {}) # Get existing part data or empty dict
            if not isinstance(part_data, dict): # Ensure it's a dict
                 part_data = {}
                 updated_data[part_name] = part_data # Replace if not a dict

            # Collect Lengths
            if part_name in self.part_length_entries:
                for length_key, entry in self.part_length_entries[part_name].items():
                    try:
                        value = float(entry.get())
                        part_data[length_key] = value
                    except ValueError:
                        validation_errors.append(f"Invalid number for {length_key} in {part_name}")
                        part_data[length_key] = 0.0 # Default or keep old value? Default for now

            # Collect Measurements
            if part_name in self.part_measurement_entries:
                measurements_list = []
                for i, (pos_entry, diam_entry) in enumerate(self.part_measurement_entries[part_name]):
                    try:
                        pos = float(pos_entry.get())
                        diam = float(diam_entry.get())
                        measurements_list.append({"position": pos, "diameter": diam})
                    except ValueError:
                        validation_errors.append(f"Invalid number for measurement {i+1} in {part_name}")
                        # Skip this measurement or add with default? Skip for now.
                # Sort measurements by position before saving
                try:
                    measurements_list.sort(key=lambda item: item.get('position', 0.0))
                except Exception as e_sort_meas_apply:
                     logger.warning(f"Could not sort measurements for part {part_name} during apply: {e_sort_meas_apply}")
                part_data['measurements'] = measurements_list


            # Collect Holes
            if part_name in self.part_hole_entries:
                hole_positions, hole_diameters, hole_chimneys, hole_diameters_out = [], [], [], []
                hole_data_for_sort = [] # List of dicts for sorting

                for i, (pos_entry, diam_entry, chimney_entry, diam_out_entry) in enumerate(self.part_hole_entries[part_name]):
                    try:
                        pos = float(pos_entry.get())
                        diam = float(diam_entry.get())

                        # Chimney height
                        try:
                            chimney_str = chimney_entry.get()
                            # Determine default based on part (headjoint embouchure vs. other holes)
                            default_chimney_mm = (DEFAULT_EMBOUCHURE_CHIMNEY_HEIGHT * M_TO_MM_FACTOR) \
                                               if part_name == FLUTE_PARTS_ORDER[0] and i == 0 \
                                               else (DEFAULT_CHIMNEY_HEIGHT * M_TO_MM_FACTOR)

                            chimney = float(chimney_str) if chimney_str.strip() else default_chimney_mm
                            if chimney <= 1e-6: # If zero or very small (e.g., user entered 0 or field was empty and default was bad)
                                chimney = default_chimney_mm
                        except ValueError:
                            chimney = default_chimney_mm # Fallback to default
                            validation_errors.append(f"Invalid chimney for hole {i+1} in {part_name}, using default {chimney:.2f}mm.")

                        # Diameter out
                        try:
                            diam_out_str = diam_out_entry.get()
                            # Default diam_out is based on inner diam
                            default_diam_out = diam * DEFAULT_HOLE_RADIUS_OUT_FACTOR
                            if default_diam_out <= 1e-6 and diam > 1e-6: default_diam_out = diam * 1.05 # Ensure slightly larger if diam is small
                            elif default_diam_out <= 1e-6 and diam <= 1e-6: default_diam_out = 0.1 # Miniscule but non-zero

                            diam_out = float(diam_out_str) if diam_out_str.strip() else default_diam_out
                            if diam_out <= 1e-6: # If zero or very small
                                diam_out = default_diam_out
                        except ValueError:
                            diam_out = default_diam_out # Fallback to default
                            validation_errors.append(f"Invalid diameter_out for hole {i+1} in {part_name}, using default {diam_out:.2f}mm.")

                        if diam <= 1e-6: # Ensure inner diameter is not zero
                            diam = 1.0 # Default to 1mm diameter if zero
                            if diam_out <= diam : diam_out = diam * DEFAULT_HOLE_RADIUS_OUT_FACTOR # Recalculate if diam_out was based on zero diam
                            validation_errors.append(f"Zero/invalid diameter for hole {i+1} in {part_name}, set to {diam:.2f}mm, diam_out to {diam_out:.2f}mm.")

                        hole_data_for_sort.append({'position': pos, 'diameter': diam, 'chimney': chimney, 'diameter_out': diam_out})
                    except ValueError:
                        validation_errors.append(f"Invalid number for hole {i+1} in {part_name}")
                        # Skip this hole or add with default? Skip for now.

                # Sort holes by position
                try:
                    hole_data_for_sort.sort(key=lambda item: item.get('position', 0.0))
                except Exception as e_sort_holes_apply:
                     logger.warning(f"Could not sort holes for part {part_name} during apply: {e_sort_holes_apply}")

                # Populate the separate lists from sorted data
                for hole_data in hole_data_for_sort:
                    hole_positions.append(hole_data.get('position', 0.0))
                    hole_diameters.append(hole_data.get('diameter', 0.0))
                    hole_chimneys.append(hole_data.get('chimney', 0.0))
                    hole_diameters_out.append(hole_data.get('diameter_out', 0.0))

                part_data['Holes position'] = hole_positions
                part_data['Holes diameter'] = hole_diameters
                part_data['Holes chimney'] = hole_chimneys
                part_data['Holes diameter_out'] = hole_diameters_out

            # Update the part data in the main updated_data dict
            updated_data[part_name] = part_data


        if validation_errors:
            error_msg = "Validation Errors:\n" + "\n".join(validation_errors)
            messagebox.showwarning("Validation Failed", error_msg, parent=self)
            # Decide whether to proceed or stop. For now, we'll proceed with potentially invalid data.
            logger.warning("Validation errors encountered, proceeding with potentially invalid data.")


        self.current_data = updated_data # Update the editor's internal state
        self._set_dirty(False) # Mark as clean after applying (before callback)

        # Call the callback function in the main app
        self.apply_callback(copy.deepcopy(self.current_data)) # Pass a copy

        # Optionally close the editor after applying
        # self._on_close() # Decide if editor should close automatically

    def _on_close(self):
        logger.info("Close button clicked.")
        if self._is_dirty:
            if messagebox.askyesno("Unsaved Changes",
                                   "You have unsaved changes in the editor.\n"
                                   "Do you want to discard them and close?",
                                   parent=self, icon=messagebox.WARNING):
                self.destroy()
            else:
                # Don't close
                pass
        else:
            self.destroy()


if __name__ == '__main__':
    # This part is for testing the editor in isolation
    root = tk.Tk()
    root.withdraw() # Hide the root window

    # Create some dummy data for testing
    dummy_data = {
        "Flute Model": "TestFlute",
        "headjoint": {
            "Total length": 200.0,
            "Mortise length": 10.0,
            "measurements": [
                {"position": 0.0, "diameter": 19.5},
                {"position": 50.0, "diameter": 19.0},
                {"position": 150.0, "diameter": 18.5}
            ],
            "Holes position": [10.0],
            "Holes diameter": [12.0],
            "Holes chimney": [5.0],
            "Holes diameter_out": [14.0]
        },
        "left": {
             "Total length": 250.0,
             "Mortise length": 10.0,
             "measurements": [
                {"position": 0.0, "diameter": 18.5},
                {"position": 100.0, "diameter": 18.0},
                {"position": 240.0, "diameter": 17.5}
            ],
            "Holes position": [20.0, 50.0, 80.0],
            "Holes diameter": [8.0, 9.0, 8.5],
            "Holes chimney": [3.0, 3.0, 3.0],
            "Holes diameter_out": [9.0, 10.0, 9.5]
        },
        "right": {
             "Total length": 200.0,
             "Mortise length": 10.0,
             "measurements": [
                {"position": 0.0, "diameter": 17.5},
                {"position": 190.0, "diameter": 17.0}
            ],
            "Holes position": [30.0, 60.0, 90.0],
            "Holes diameter": [7.0, 7.5, 7.0],
            "Holes chimney": [3.0, 3.0, 3.0],
            "Holes diameter_out": [8.0, 8.5, 8.0]
        },
        "foot": {
             "Total length": 100.0,
             "Mortise length": 10.0,
             "measurements": [
                {"position": 0.0, "diameter": 17.0},
                {"position": 90.0, "diameter": 16.5}
            ],
            "Holes position": [20.0, 50.0],
            "Holes diameter": [6.0, 6.5],
            "Holes chimney": [3.0, 3.0],
            "Holes diameter_out": [7.0, 7.5]
        }
    }

    def test_callback(updated_data: Dict[str, Any]):
        print("\nCallback received updated data:")
        # print(json.dumps(updated_data, indent=4))
        print(f"Flute Model: {updated_data.get('Flute Model')}")
        print(f"Headjoint measurements count: {len(updated_data.get('headjoint', {}).get('measurements', []))}")
        print(f"Left holes count: {len(updated_data.get('left', {}).get('Holes position', []))}")
        # You can add more checks here

    # Check if FluteData and FluteOperations can be imported for plotting
    try:
        _ = FluteData # Check if class exists
        _ = FluteOperations # Check if class exists
        logger.info("FluteData and FluteOperations imported successfully for editor test.")
        editor = GraphicalFluteEditor(root, dummy_data, "TestFlute", test_callback)
        root.mainloop()
    except ImportError as e:
        logger.error(f"Could not import necessary modules for editor test: {e}")
        print(f"Error: Could not import necessary modules for editor test: {e}")
        print("Please ensure flute_data.py, flute_operations.py, and constants.py are in the same directory.")
    except Exception as e:
        logger.exception("An error occurred during the editor test:")
        print(f"An unexpected error occurred during the editor test: {e}")