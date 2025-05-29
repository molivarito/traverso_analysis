import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from openwind import ImpedanceComputation, Player, InstrumentGeometry
import numpy as np
from matplotlib.cm import tab10
import logging
from typing import Any, List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)

MM_TO_M_FACTOR = 1e-3
M_TO_MM_FACTOR = 1e3
DEFAULT_FLUTE_PART_ORDER = ["headjoint", "left", "right", "foot"]
class FluteOperations:
    def __init__(self, flute_data: Any) -> None:
        """
        Constructor for the FluteOperations class.

        Args:
            flute_data (Any): Instance containing flute data.
        """
        self.flute_data = flute_data
        self.flute_color: str = self._assign_color(flute_data.data.get("Flute Model", "Default"))

    @staticmethod
    def _assign_color(flute_name: str) -> str:
        """
        Assigns a unique color based on the flute name.

        Args:
            flute_name (str): The flute's name.

        Returns:
            str: A hexadecimal color code.
        """
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        index = hash(flute_name) % len(colors)
        return colors[index]

    def _calculate_adjusted_positions(self, part: str, current_position: float) -> Tuple[List[float], List[float]]:
        """
        Calculates adjusted positions for a given flute part based on the current offset.

        Args:
            part (str): The flute part (e.g., 'headjoint').
            current_position (float): The current offset.

        Returns:
            Tuple[List[float], List[float]]: The adjusted positions and diameters.
        """
        positions = [item["position"] for item in self.flute_data.data[part].get("measurements", [])]
        diameters = [item["diameter"] for item in self.flute_data.data[part].get("measurements", [])]
        adjusted_positions = [pos + current_position for pos in positions]
        return adjusted_positions, diameters

    def plot_individual_parts(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                              flute_color: Optional[str] = None) -> Tuple[plt.Figure, List[Any]]:
        """
        Plots each part of the flute separately along with its holes.

        Args:
            ax (Optional[Any]): Axes to plot on. If None, new axes are created.
            flute_names (Optional[List[str]]): List of flute names for the title.
            flute_color (Optional[str]): Color to use for plotting.

        Returns:
            Tuple[plt.Figure, List[Any]]: The figure and list of axes.
        """
        part_order = DEFAULT_FLUTE_PART_ORDER
        linestyles = ['-', '--', '-.', ':']

        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            ax = axes.flatten()
        elif isinstance(ax, plt.Axes):
            fig = ax.figure
            ax = [ax]
        else:
            fig = ax[0].figure

        if len(ax) < len(part_order):
            # Si solo se proporciona un eje, y se necesitan más, esto fallará.
            # Considerar crear nuevos ejes si el 'ax' proporcionado no es suficiente, o lanzar un error más claro.
            # Por ahora, se asume que si se pasa 'ax', es una lista de ejes del tamaño adecuado.
            pass

        for i, part in enumerate(part_order):
            adjusted_positions, diameters = self._calculate_adjusted_positions(part, 0)
            linestyle = linestyles[i % len(linestyles)]
            # El label se asigna directamente, la leyenda del subplot se encargará de no duplicar si se llama múltiples veces con el mismo label.
            label = self.flute_data.data.get("Flute Model", "Unknown") if flute_names else None
            
            if isinstance(ax, (list, np.ndarray)): # Si ax es una lista o array de ejes
                current_ax = ax[i]
            else: # Si ax es un solo objeto Axes
                current_ax = ax
            current_ax.plot(adjusted_positions, diameters, marker='o', linestyle=linestyle,
                       color=flute_color, markersize=4, label=label)

            # Plot holes without adding duplicate legend entries
            hole_positions = self.flute_data.data[part].get("Holes position", [])
            hole_diameters = self.flute_data.data[part].get("Holes diameter", [])
            if hole_positions and hole_diameters:
                for pos, diam in zip(hole_positions, hole_diameters):
                    current_ax.plot(pos, 10, color=flute_color, marker='o', markersize=diam * 2) # Y-position for holes is arbitrary

            current_ax.set_xlabel("Position (mm)")
            current_ax.set_ylabel("Diameter (mm)")
            current_ax.set_title(f"{part.capitalize()} Geometry")
            current_ax.grid(True)
            if label: current_ax.legend(loc='best', fontsize=8) # 'best' o 'upper right'

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if flute_names:
            fig.suptitle(", ".join(flute_names), fontsize=12)

        return fig, ax

    def plot_all_parts_overlapping(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                                   flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> Any:
        """
        Plots all parts of the flute overlapping to show the complete profile.

        Args:
            ax (Optional[Any]): Axis to plot on. A new axis is created if None.
            flute_names (Optional[List[str]]): List of flute names for the title.
            flute_color (Optional[str]): Color to use.
            flute_style (Optional[str]): Line style to use.

        Returns:
            Any: The axis with the plotted data.
        """
        if ax is None:
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)

        part_order = DEFAULT_FLUTE_PART_ORDER
        current_position = 0

        for i, part in enumerate(part_order):
            adjusted_positions, diameters = self._calculate_adjusted_positions(part, current_position)
            label = self.flute_data.data.get("Flute Model", "Unknown") if i == 0 and flute_names else None
            ax.plot(adjusted_positions, diameters, marker='o', linestyle=flute_style,
                    color=flute_color, markersize=4, label=label)
      
            hole_positions = self.flute_data.data[part].get("Holes position", [])
            hole_diameters = self.flute_data.data[part].get("Holes diameter", [])
            if hole_positions and hole_diameters:
                for pos, diam in zip(hole_positions, hole_diameters):
                    ax.plot(pos + current_position, 10, color=flute_color, marker='o', markersize=diam * 2)

            total_length = self.flute_data.data[part].get("Total length", 0)
            current_position += total_length

        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Diameter (mm)")
        ax.legend(loc='upper right')
        ax.grid(True)
        return ax

    def plot_combined_flute_data(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                                 flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> Any:
        """
        Plots the combined flute profile using measurements from all parts.

        Args:
            ax (Optional[Any]): Axis to plot on. A new axis is created if None.
            flute_names (Optional[List[str]]): List of flute names.
            flute_color (Optional[str]): Color to use.
            flute_style (Optional[str]): Line style to use.

        Returns:
            Any: The axis with the plotted data.
        """
        if ax is None:
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)

        combined_measurements = self.flute_data.combine_measurements()
        positions = [item["position"] for item in combined_measurements]
        diameters = [item["diameter"] for item in combined_measurements]

        ax.plot(positions, diameters, label=self.flute_data.data.get("Flute Model", "Unknown"),
                linestyle=flute_style, color=flute_color)

        part_order = DEFAULT_FLUTE_PART_ORDER
        current_position = 0

        for part in part_order:
            hole_positions = self.flute_data.data[part].get("Holes position", [])
            hole_diameters = self.flute_data.data[part].get("Holes diameter", [])
            if hole_positions and hole_diameters:
                hole_positions = [pos + current_position for pos in hole_positions]
                for pos, diam in zip(hole_positions, hole_diameters):
                    ax.plot(pos, 10, color=flute_color, marker='o', markersize=diam * 2)
            current_position += self.flute_data.data[part].get("Total length", 0) - \
                                self.flute_data.data[part].get("Mortise length", 0)

        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Diameter (mm)")
        ax.grid(True)
        ax.legend(loc='upper right')
        return ax

    def plot_flute_2d_view(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                           flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> Any:
        """
        Plots a simplified 2D view of the flute based on combined measurements.

        Args:
            ax (Optional[Any]): Axis to plot on. A new axis is created if None.
            flute_names (Optional[List[str]]): List of flute names.
            flute_color (Optional[str]): Color to use.
            flute_style (Optional[str]): Line style to use.

        Returns:
            Any: The axis with the plotted data.
        """
        if ax is None:
            fig = plt.figure(figsize=(15, 3))
            ax = fig.add_subplot(111)

        combined_measurements = self.flute_data.combine_measurements()
        positions = [item["position"] for item in combined_measurements]
        diameters = [item["diameter"] for item in combined_measurements]

        ax.plot(positions, [d / 2 for d in diameters], color=flute_color, linestyle=flute_style,
                linewidth=2, label=self.flute_data.data.get("Flute Model", "Unknown") if flute_names else "Flute")
        ax.plot(positions, [-d / 2 for d in diameters], color=flute_color, linestyle=flute_style, linewidth=2)

        part_order = DEFAULT_FLUTE_PART_ORDER
        current_position = 0

        for part in part_order:
            hole_positions = self.flute_data.data[part].get("Holes position", [])
            hole_diameters = self.flute_data.data[part].get("Holes diameter", [])
            if hole_positions and hole_diameters:
                hole_positions = [pos + current_position for pos in hole_positions]
                for pos, diam in zip(hole_positions, hole_diameters):
                    scaled_markersize = max(diam * 0.5, 2)
                    ax.plot(pos, 0, color=flute_color, marker='o', markersize=scaled_markersize)
            current_position += self.flute_data.data[part].get("Total length", 0)

        ax.set_xlabel("Position (mm)")
        ax.set_ylabel("Radius (mm)")
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='upper right')
        ax.grid(True)
        return ax

    def plot_instrument_geometry(self, note: str = "D", ax: Optional[Any] = None) -> Optional[Any]:
        """
        Plots the instrument geometry using the InstrumentGeometry data from the acoustic analysis.

        Args:
            note (str, optional): The note for which to obtain geometry. Defaults to "D".
            ax (Optional[Any]): Axis to plot on. A new axis is created if None.

        Returns:
            Optional[Any]: The axis with the plotted data, or None if an error occurs.
        """
        try:
            acoustic_analysis = self.flute_data.acoustic_analysis[note]
            # isinstance check is more Pythonic than hasattr for methods if you know the expected type
            if not isinstance(acoustic_analysis, ImpedanceComputation):
                 logger.warning(f"Acoustic analysis for note '{note}' is not of type ImpedanceComputation.")
                 # Or raise TypeError if it's a critical issue
                 # return None 

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6))
            acoustic_analysis.plot_instrument_geometry(ax=ax)
            ax.set_title(f"Instrument Geometry for {note} - {self.flute_data.data['Flute Model']}")
            ax.set_xlabel("Position")
            ax.set_ylabel("Radius")
            ax.grid(True)
            return ax
        except Exception as e:
            logger.error(f"Error plotting instrument geometry for note {note}: {e}")
            return None

    # Consider making this public if called from GUI/data_processing
    def plot_individual_admittance_analysis(self, acoustic_analysis_list: List[Tuple[Any, str]], note: str) -> plt.Figure:
        """
        Creates a multi-panel figure showing admittance, pressure, geometry, and flow for a specific note.

        Args:
            acoustic_analysis_list (List[Tuple[Any, str]]): List of tuples (acoustic analysis, flute name).
            note (str): The note to plot.

        Returns:
            plt.Figure: The generated figure.
        """
        linestyles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        fig, axes = plt.subplots(4, 1, figsize=(20, 30), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        ax_admittance, ax_pressure, ax_geometry, ax_flow = axes

        for index, (analysis, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = colors[index % len(colors)]
            if note in analysis:
                frequencies = analysis[note].frequencies
                impedance = analysis[note].impedance
                admittance = 1 / impedance

                ax_admittance.plot(frequencies, 20 * np.log10(np.abs(admittance)),
                                   linestyle=linestyle, color=color, label=flute_name)

                antiresonant_frequencies = list(analysis[note].antiresonance_frequencies())
                ymin, ymax = ax_admittance.get_ylim()
                for i, f in enumerate(antiresonant_frequencies[:5]):
                    ax_admittance.vlines(f, ymin, ymax, color=color, linestyle=':', alpha=0.7)
                    ax_admittance.text(f, ymin + (ymax - ymin) * 0.95, f"$\\bf{{{f:.1f} Hz}}$",
                                       rotation=90, color=color, fontsize=10, ha='center', va='bottom')

                x, pressure, flow = analysis[note].get_pressure_flow()
                pressure = np.abs(pressure.T)
                flow = np.abs(flow.T)
                idx_f1 = np.argmin(np.abs(frequencies - antiresonant_frequencies[0]))
                idx_f2 = np.argmin(np.abs(frequencies - antiresonant_frequencies[1]))

                ax_pressure.plot(x, pressure[:, idx_f1], linestyle='-', color=color,
                                 label=f"{flute_name} - {antiresonant_frequencies[0]:.1f} Hz")
                ax_pressure.plot(x, pressure[:, idx_f2], linestyle='--', color=color,
                                 label=f"{flute_name} - {antiresonant_frequencies[1]:.1f} Hz")
                ax_flow.plot(x, flow[:, idx_f1], linestyle='-', color=color,
                            label=f"{flute_name} - {antiresonant_frequencies[0]:.1f} Hz")
                ax_flow.plot(x, flow[:, idx_f2], linestyle='--', color=color,
                            label=f"{flute_name} - {antiresonant_frequencies[1]:.1f} Hz")
                try:
                    instrument_geometry = analysis[note].get_instrument_geometry()
                    if instrument_geometry:
                        for shape in instrument_geometry.main_bore_shapes:
                            self._plot_shape(shape, ax_geometry, mmeter=M_TO_MM_FACTOR, color=color, linewidth=1)
                        self._plot_holes(instrument_geometry.holes, ax_geometry, mmeter=M_TO_MM_FACTOR, note=note, color=color, acoustic_analysis_obj=analysis[note])
                        ax_geometry.set_aspect('equal', adjustable='datalim')
                except Exception as e:
                    logger.error(f"Error plotting geometry: {e}")

        ax_admittance.set_title(f"Admittance for {note}")
        ax_admittance.set_xlabel("Frequency (Hz)")
        ax_admittance.set_ylabel("Admittance (dB)")
        ax_admittance.legend(loc='upper right')
        ax_admittance.grid(True)

        ax_pressure.set_title(f"Pressure vs Position for {note}")
        ax_pressure.set_xlabel("Position (m)")
        ax_pressure.set_ylabel("Pressure (Pa)")
        ax_pressure.legend(loc='upper right')
        ax_pressure.grid(True)

        ax_geometry.set_title(f"Top View - Instrument Geometry for {note}")
        ax_geometry.set_xlabel("Position (mm)")
        ax_geometry.set_ylabel("Radius (mm)")
        ax_geometry.grid(True)

        ax_flow.set_title(f"Flow vs Position for {note}")
        ax_flow.set_xlabel("Position (m)")
        ax_flow.set_ylabel("Flow (m³/s)")
        ax_flow.legend(loc='upper right')
        ax_flow.grid(True)

        fig.tight_layout()
        return fig

    # Consider making this public
    def plot_combined_admittance(self, acoustic_analysis_list: List[Tuple[Any, str]]) -> plt.Figure:
        """
        Generates a combined admittance plot for all notes from multiple flutes.

        Args:
            acoustic_analysis_list (List[Tuple[Any, str]]): List of tuples (acoustic analysis, flute name).

        Returns:
            plt.Figure: The generated figure.
        """
        linestyles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        legend_items = []

        for index, (analysis, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = colors[index % len(colors)]
            for note in analysis.keys():
                analysis[note].plot_admittance(figure=fig, linestyle=linestyle, color=color)
            legend_items.append((linestyle, color, flute_name))
        legend_handles = [plt.Line2D([0], [0], color=color, linestyle=ls, label=flute)
                          for ls, color, flute in legend_items]
        ax.legend(handles=legend_handles, loc='upper right')
        ax.set_title("Combined Admittance for All Notes")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Admittance")
        ax.grid(True)
        return fig

    # Consider making this public
    def plot_summary_antiresonances(self, acoustic_analysis_list: List[Tuple[Any, str]], notes: List[str]) -> plt.Figure:
        """
        Plots antiresonant frequencies for each note across multiple flutes, including labels for the first two values.

        Args:
            acoustic_analysis_list (List[Tuple[Any, str]]): List of tuples (acoustic analysis, flute name).
            notes (List[str]): List of notes to plot.

        Returns:
            plt.Figure: The generated figure.
        """
        linestyles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        legend_items = []
        note_positions = {note: i for i, note in enumerate(notes)}
        offset = 0.2

        for index, (analysis, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = colors[index % len(colors)]
            for note in notes:
                if note in analysis:
                    freqs = list(analysis[note].antiresonance_frequencies())
                    if freqs:
                        x_positions = [note_positions[note] + index * offset for _ in freqs]
                        ax.plot(x_positions, freqs, "o", linestyle=linestyle, color=color)
                        for i, (x, f) in enumerate(zip(x_positions, freqs)):
                            if i < 2:
                                ax.text(x, f, f"{f:.1f} Hz", fontsize=10, ha="center", va="bottom",
                                        rotation=45, alpha=0.7)
            legend_items.append((linestyle, color, flute_name))
        ax.set_xticks(range(len(notes)))
        ax.set_xticklabels(notes)
        legend_handles = [plt.Line2D([0], [0], color=color, linestyle=ls, label=flute)
                          for ls, color, flute in legend_items]
        ax.legend(handles=legend_handles, loc='upper right')
        ax.set_title("Antiresonant Frequencies vs. Note")
        ax.set_xlabel("Note")
        ax.set_ylabel("Frequency (Hz)")
        ax.grid(True)
        return fig

    # Consider making this public
    def plot_summary_cents_differences(self, acoustic_analysis_list: List[Tuple[Any, str]], notes: List[str]) -> plt.Figure:
        """
        Plots the difference in cents between peaks for each note across multiple flutes.

        Args:
            acoustic_analysis_list (List[Tuple[Any, str]]): List of tuples (acoustic analysis, flute name).
            notes (List[str]): List of notes to plot.

        Returns:
            plt.Figure: The generated figure.
        """
        linestyles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        legend_items = []

        for index, (analysis, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = colors[index % len(colors)]
            cents_differences = []
            for note in notes:
                if note in analysis:
                    antiresonant_frequencies = list(analysis[note].antiresonance_frequencies())
                    if len(antiresonant_frequencies) >= 2:
                        f1, f2 = antiresonant_frequencies[:2]
                        cents_differences.append(1200 * np.log2(f2 / (2 * f1)))
                    else:
                        cents_differences.append(None)
            ax.plot(notes, cents_differences, "o", linestyle=linestyle, color=color)
            legend_items.append((linestyles[index % len(linestyles)], color, flute_name))
        legend_handles = [plt.Line2D([0], [0], color=color, linestyle=ls, label=flute)
                          for ls, color, flute in legend_items]
        ax.legend(handles=legend_handles, loc='upper right')
        ax.set_title("Inharmonicity in Cents (Second Peak - 2x First Peak)")
        ax.set_xlabel("Note")
        ax.set_ylabel("Difference (cents)")
        ax.grid(True)
        return fig

    def _plot_shape(self, shape: Any, ax: Any, mmeter: float, shift_x: float = 0, shift_y: float = 0, **kwargs: Any) -> None:
        """
        Draws the main shape of the tube in the top view.

        Args:
            shape (Any): A segment of the main bore geometry.
            ax (Any): The axis to plot on.
            mmeter (float): Scale factor to convert to millimeters.
            shift_x (float): Horizontal offset.
            shift_y (float): Vertical offset.
            **kwargs: Additional keyword arguments for plotting.
        """
        x, r = InstrumentGeometry._get_xr_shape(shape)
        radius = np.append(r, np.nan)
        position = np.append(x, np.nan) + shift_x
        ax.plot(np.append(position, np.flip(position)) * mmeter,
                (np.append(radius, np.flip(-radius)) + shift_y) * mmeter,
                **kwargs)

    def _plot_holes(self, holes: List[Any], ax: Any, mmeter: float, note: Optional[str] = None, acoustic_analysis_obj: Optional[ImpedanceComputation] = None, **kwargs: Any) -> None:
        """
        Draws the holes on the top view.

        Args:
            holes (List[Any]): List of holes.
            ax (Any): The axis to plot on.
            acoustic_analysis_obj (ImpedanceComputation): The acoustic analysis object for the current note and flute.
            mmeter (float): Scale factor to convert to millimeters.
            note (Optional[str]): Specific note to determine open/closed state.
            **kwargs: Additional keyword arguments for plotting.
        """
        try:
            theta = np.linspace(0, 2 * np.pi, 100)
            for hole in holes:
                position = hole.position.get_value()
                radius = np.mean(hole.shape.get_radius_at(np.linspace(0, 1, 10)))
                x_circle = position + radius * np.cos(theta)
                y_circle = radius * np.sin(theta)
                # Determine whether to use fill or plot based on fingering info
                if note and acoustic_analysis_obj:
                    # Ensure acoustic_analysis_obj is the one for the specific flute being plotted, not self.flute_data...
                    # if this helper is used for multiple flutes on the same axes.
                    # The current call from plot_individual_admittance_analysis passes the correct analysis object.
                    fingering = acoustic_analysis_obj.get_instrument_geometry().fingering_chart.fingering_of(note)
                    plot_func = ax.plot if fingering.is_side_comp_open(hole.label) else ax.fill
                else:
                    plot_func = ax.plot
                hole_plot = plot_func(x_circle * mmeter, y_circle * mmeter, **kwargs)
                if hole_plot and hasattr(hole_plot[0], "set_edgecolor"):
                    hole_plot[0].set_edgecolor(hole_plot[0].get_facecolor())
        except Exception as e:
            logger.error(f"Error plotting holes: {e}")

    def plot_top_view_instrument_geometry(self, note: str = "D") -> Optional[plt.Figure]:
        """
        Generates the top view of the instrument geometry with holes for a specific note.

        Args:
            note (str): The specific note for which to determine open/closed holes.

        Returns:
            Optional[plt.Figure]: The generated figure, or None if an error occurs.
        """
        try:
            # mmeter = M_TO_MM_FACTOR # Already defined as argument
            fig, ax = plt.subplots(figsize=(15, 3))
            ax.set_aspect('equal', adjustable='datalim')
            instrument_geometry = self.flute_data.acoustic_analysis[note].get_instrument_geometry()
            if not instrument_geometry:
                raise ValueError("Could not obtain instrument geometry.")
            logger.info("Plotting main tube shape...")
            for shape in instrument_geometry.main_bore_shapes:
                self._plot_shape(shape, ax, M_TO_MM_FACTOR, shift_x=0, shift_y=0, color='black', linewidth=1)
            self._plot_holes(instrument_geometry.holes, ax, M_TO_MM_FACTOR, note, acoustic_analysis_obj=self.flute_data.acoustic_analysis[note])
            ax.set_xlabel("Position (mm)")
            ax.set_ylabel("Radius (mm)")
            ax.set_title(f"Top View - Instrument Geometry for Note '{note}'")
            ax.grid(True)
            return fig
        except Exception as e:
            logger.error(f"Error generating top view geometry: {e}")
            return None

    def plot_moc_summary(self, acoustic_analysis_list: List[Tuple[Any, str]], finger_frequencies: Dict[str, float], notes: Optional[List[str]] = None) -> plt.Figure:
        """
        Generates a summary plot of the MOC value for each note across multiple flutes.
        The MOC is defined as (f1 - f0 - fplay) / ((f0 - fplay) * (f1 - fplay)),
        where fplay is the frequency of the note from the finger chart, f0 is the first antiresonance frequency,
        and f1 is the second antiresonance frequency.
        
        Args:
            acoustic_analysis_list (List[Tuple[Any, str]]): List of tuples (acoustic analysis, flute name).
            finger_frequencies (Dict[str, float]): Dictionary with note frequencies (fplay) from the finger chart.
            notes (Optional[List[str]]): List of notes to include. If None, uses the keys from finger_frequencies.
        
        Returns:
            plt.Figure: The generated figure.
        """
        import numpy as np
        if notes is None:
            notes = list(finger_frequencies.keys())
        linestyles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig, ax = plt.subplots(figsize=(10, 6))
        # Map each note to an x-position (0, 1, 2, ...)
        note_positions = {note: i for i, note in enumerate(notes)}
        offset = 0.1  # Offset to separate points from different flutes
        for index, (analysis, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = colors[index % len(colors)]
            moc_values = []
            x_positions = []
            for note in notes:
                try:
                    if note in analysis:
                        antiresonances = list(analysis[note].antiresonance_frequencies())
                    if len(antiresonances) >= 2 and finger_frequencies.get(note) is not None:
                        f0 = antiresonances[0]
                        f1 = antiresonances[1]
                        fplay = finger_frequencies[note]
                        if f0 != 0 and f1 != 0 and fplay != 0:
                            numerator = (1 / f1) - (1 / (2 * fplay))
                            denominator = (1 / f0) - (1 / fplay)
                            moc = numerator / denominator if denominator != 0 else np.nan
                        else:
                            moc = np.nan
                    else:
                        moc = np.nan
                except Exception as e:
                    logger.error(f"Error computing MOC for note {note} in flute {flute_name}: {e}")
                    moc = np.nan
                moc_values.append(moc)
                x_positions.append(note_positions[note])
            ax.plot(x_positions, moc_values, "o", linestyle=linestyle, color=color, label=flute_name)
        ax.set_xticks(range(len(notes)))
        ax.set_xticklabels(notes)
        ax.set_xlabel("Nota")
        ax.set_ylabel("MOC (ratio)")
        ax.set_title("Gráfico de MOC por Nota")
        ax.grid(True)
        ax.legend(loc='upper right')
        return fig
    
    def plot_bi_espe_summary(self, acoustic_analysis_list: List[Tuple[Any, str]], finger_frequencies: Dict[str, float], notes: Optional[List[str]] = None, speed_of_sound: float = 343.0) -> plt.Figure:
        if notes is None:
            notes = list(finger_frequencies.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        linestyles = ['-', '--', '-.', ':']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for idx, (analysis, flute_name) in enumerate(acoustic_analysis_list):
            bi_values = []
            espe_values = []
            note_positions = []
            for note in notes:
                try:
                    fplay_I = finger_frequencies[note]
                    fplay_II = 2 * fplay_I

                    if note not in analysis:
                        bi_values.append(np.nan)
                        espe_values.append(np.nan)
                        continue

                    antiresonances = list(analysis[note].antiresonance_frequencies())
                    if len(antiresonances) < 2:
                        bi_values.append(np.nan)
                        espe_values.append(np.nan)
                        continue

                    f0 = antiresonances[0]
                    f1 = antiresonances[1]

                    # First-Octave Pitch Adjustment (B_I)
                    bi = 1200 * np.log2(fplay_I / f0)

                    # Embouchure corrections
                    delta_l_I = (speed_of_sound / 2) * (1 / fplay_I - 1 / f0)
                    delta_l_II = speed_of_sound * (1 / fplay_II - 1 / f1)
                    delta_delta_l = delta_l_II - delta_l_I
                    L_eff_I = speed_of_sound / (2 * fplay_I)

                    # ESPE
                    if L_eff_I + delta_delta_l > 0:
                        espe = 1200 * np.log2(L_eff_I / (L_eff_I + delta_delta_l))
                    else:
                        espe = np.nan

                    bi_values.append(bi)
                    espe_values.append(espe)
                    note_positions.append(note)

                except Exception as e:
                    logger.error(f"Error computing B_I or ESPE for note {note}: {e}")
                    bi_values.append(np.nan)
                    espe_values.append(np.nan)

            ax.plot(note_positions, bi_values, label=f"{flute_name} - $B_I$", linestyle='-', color=colors[idx % len(colors)], marker='o')
            ax.plot(note_positions, espe_values, label=f"{flute_name} - ESPE", linestyle='--', color=colors[idx % len(colors)], marker='x')

        ax.set_title("$B_I$ and ESPE across notes")
        ax.set_xlabel("Note")
        ax.set_ylabel("Cents")
        ax.legend()
        ax.grid(True)
        return fig

    def plot_summary_pdf(self, acoustic_analysis_list: List[Tuple[Any, str]], finger_frequencies: Dict[str, float], notes: Optional[List[str]] = None, pdf_filename: str = "summary.pdf") -> str:
        """
        Generates a PDF report including the MOC summary and the B_I/ESPE summary.
 
        Args:
            acoustic_analysis_list (List[Tuple[Any, str]]): List of tuples (acoustic analysis, flute name).
            finger_frequencies (Dict[str, float]): Dictionary with note frequencies.
            notes (Optional[List[str]]): List of notes to include.
            pdf_filename (str): The filename for the generated PDF.
 
        Returns:
            str: The path to the generated PDF.
        """
        with PdfPages(pdf_filename) as pdf:
            fig_moc = self.plot_moc_summary(acoustic_analysis_list, finger_frequencies, notes)
            pdf.savefig(fig_moc)
            plt.close(fig_moc)

            fig_bi_espe = self.plot_bi_espe_summary(acoustic_analysis_list, finger_frequencies, notes)
            pdf.savefig(fig_bi_espe)
            plt.close(fig_bi_espe)
        return pdf_filename