import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import os # No se usa actualmente
from openwind import ImpedanceComputation, Player, InstrumentGeometry # type: ignore
import numpy as np
# from matplotlib.cm import tab10 # No se usa directamente
import logging
from typing import Any, List, Tuple, Optional, Dict

from constants import (
    BASE_COLORS, LINESTYLES, FLUTE_PARTS_ORDER,
    M_TO_MM_FACTOR
)
# Necesitas FluteData aquí si FluteOperations lo usa como tipo, pero solo se pasa como 'Any' en __init__
# from flute_data import FluteData # Descomentar si se usa FluteData como tipo explícito

logger = logging.getLogger(__name__)

class FluteOperations:
    def __init__(self, flute_data_instance: Any) -> None: # flute_data_instance es una instancia de FluteData
        self.flute_data = flute_data_instance

    def _calculate_adjusted_positions(self, part: str, current_position: float) -> Tuple[List[float], List[float]]:
        # Asegurarse que self.flute_data.data[part] existe y tiene 'measurements'
        part_data = self.flute_data.data.get(part, {})
        measurements = part_data.get("measurements", [])
        positions = [item.get("position", 0.0) for item in measurements]
        diameters = [item.get("diameter", 0.0) for item in measurements]
        adjusted_positions = [pos + current_position for pos in positions]
        return adjusted_positions, diameters

    def plot_individual_parts(self, axes_list: Optional[List[plt.Axes]] = None,
                              figure_title: Optional[str] = None,
                              flute_color: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:

        fig: plt.Figure
        ax_list_to_plot_on: List[plt.Axes]

        if axes_list is None:
            fig, axes_array = plt.subplots(2, 2, figsize=(10, 8))
            ax_list_to_plot_on = list(axes_array.flatten())
        elif isinstance(axes_list, list) and all(isinstance(ax_item, plt.Axes) for ax_item in axes_list) :
            if not axes_list: # Lista vacía de ejes
                 logger.warning("Se proporcionó una lista de ejes vacía a plot_individual_parts. Creando figura por defecto.")
                 fig, axes_array = plt.subplots(2, 2, figsize=(10, 8)); ax_list_to_plot_on = list(axes_array.flatten())
            else:
                fig = axes_list[0].figure
                ax_list_to_plot_on = axes_list
        else:
             logger.warning(f"Argumento 'axes_list' ({type(axes_list)}) inesperado en plot_individual_parts, creando figura por defecto.")
             fig, axes_array = plt.subplots(2, 2, figsize=(10, 8)); ax_list_to_plot_on = list(axes_array.flatten())


        actual_flute_name = self.flute_data.flute_model

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            if i >= len(ax_list_to_plot_on):
                logger.warning(f"No hay suficientes ejes para la parte '{part_name}' en plot_individual_parts.")
                break
            current_ax = ax_list_to_plot_on[i]
            current_ax.clear()

            adjusted_positions, diameters = self._calculate_adjusted_positions(part_name, 0)

            linestyle = LINESTYLES[i % len(LINESTYLES)]
            color_to_use = flute_color if flute_color else BASE_COLORS[0]

            current_ax.plot(adjusted_positions, diameters, marker='o', linestyle=linestyle,
                       color=color_to_use, markersize=4, label=actual_flute_name)

            part_data = self.flute_data.data.get(part_name, {})
            hole_positions = part_data.get("Holes position", [])
            hole_diameters = part_data.get("Holes diameter", [])
            if hole_positions and hole_diameters:
                y_pos_for_holes = min(diameters) - 5 if diameters else -5
                for pos, diam in zip(hole_positions, hole_diameters):
                    current_ax.plot(pos, y_pos_for_holes, color=color_to_use,
                                    marker='o', markersize=max(diam * 0.5, 2), linestyle='None')

            current_ax.set_xlabel("Posición (mm)")
            current_ax.set_ylabel("Diámetro (mm)")
            current_ax.set_title(f"{part_name.capitalize()} ({actual_flute_name})", fontsize=9)
            current_ax.grid(True, linestyle=':', alpha=0.7)
            current_ax.legend(loc='best', fontsize=8)

        fig.tight_layout(rect=[0, 0.03, 1, 0.93])

        if figure_title:
            fig.suptitle(figure_title, fontsize=12)
        elif len(ax_list_to_plot_on) < len(FLUTE_PARTS_ORDER) and not figure_title:
             pass
        else:
             fig.suptitle(f"Detalle de Partes: {actual_flute_name}", fontsize=12)

        return fig, ax_list_to_plot_on

    def plot_all_parts_overlapping(self, ax: Optional[plt.Axes] = None,
                                   plot_label: Optional[str] = None,
                                   flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> plt.Axes:
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))
        else:
            fig = ax.figure
            ax.clear() # Limpiar el eje si se reutiliza

        current_position = 0.0
        actual_flute_name = self.flute_data.flute_model
        label_to_use = plot_label if plot_label else actual_flute_name

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            part_data = self.flute_data.data.get(part_name, {})
            if not part_data: continue

            adjusted_positions, diameters = self._calculate_adjusted_positions(part_name, current_position)
            # Solo etiquetar la primera parte para la leyenda general de esta flauta
            current_part_label = label_to_use if i == 0 else None

            ax.plot(adjusted_positions, diameters, marker='o',
                    linestyle=flute_style if flute_style else LINESTYLES[0],
                    color=flute_color if flute_color else BASE_COLORS[0],
                    markersize=4, label=current_part_label)

            hole_positions = part_data.get("Holes position", [])
            hole_diameters = part_data.get("Holes diameter", [])
            if hole_positions and hole_diameters: # No dibujar si no hay diámetros de tubo
                y_pos_for_holes = min(diameters) - 5 if diameters else -5
                for pos, diam in zip(hole_positions, hole_diameters):
                    ax.plot(pos + current_position, y_pos_for_holes,
                            color=flute_color if flute_color else BASE_COLORS[0],
                            marker='o', markersize=max(diam * 0.5, 2), linestyle='None')

            total_length = part_data.get("Total length", 0.0)
            current_position += total_length

        ax.set_xlabel("Posición Acumulada (mm)")
        ax.set_ylabel("Diámetro (mm)")
        if label_to_use: ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_title("Partes Desplegadas Secuencialmente" + (f" - {label_to_use}" if label_to_use else ""))
        return ax

    def plot_combined_flute_data(self, ax: Optional[plt.Axes] = None,
                                 plot_label: Optional[str] = None, # Renombrado de flute_names
                                 flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> plt.Axes:
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))
        else:
            fig = ax.figure
            # ax.clear() # Eliminado: El llamador es responsable de limpiar los ejes si es necesario antes de superponer.

        combined_measurements = self.flute_data.combined_measurements
        if not combined_measurements:
            logger.warning(f"No hay mediciones combinadas para {self.flute_data.flute_model} en plot_combined_flute_data.")
            ax.text(0.5, 0.5, "No hay datos de mediciones combinadas", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Perfil de Flauta Combinado" + (f" - {self.flute_data.flute_model}" if self.flute_data.flute_model else ""))
            return ax

        positions = [item["position"] for item in combined_measurements]
        diameters = [item["diameter"] for item in combined_measurements]

        label_to_use = plot_label if plot_label else self.flute_data.flute_model

        ax.plot(positions, diameters, label=label_to_use,
                linestyle=flute_style if flute_style else LINESTYLES[0],
                color=flute_color if flute_color else BASE_COLORS[0])

        # TODO: Ploteo de agujeros en posiciones absolutas si es necesario y los datos están disponibles.
        # Ver plot_top_view_instrument_geometry para una implementación de esto.

        ax.set_xlabel("Posición (mm)")
        ax.set_ylabel("Diámetro (mm)")
        ax.grid(True, linestyle=':', alpha=0.7)
        if label_to_use: ax.legend(loc='best', fontsize=9)
        title_str = "Perfil de Flauta Combinado"
        if self.flute_data.flute_model and self.flute_data.flute_model != label_to_use:
            title_str += f" ({self.flute_data.flute_model})"
        ax.set_title(title_str)
        return ax

    def plot_flute_2d_view(self, ax: Optional[plt.Axes] = None,
                           plot_label: Optional[str] = None, # Renombrado
                           flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> plt.Axes:
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 4))
        else:
            fig = ax.figure
            ax.clear()

        combined_measurements = self.flute_data.combined_measurements
        if not combined_measurements:
            logger.warning(f"No hay mediciones combinadas para {self.flute_data.flute_model} en plot_flute_2d_view.")
            ax.text(0.5, 0.5, "No hay datos de mediciones combinadas", ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Vista 2D de la Flauta" + (f" - {self.flute_data.flute_model}" if self.flute_data.flute_model else ""))
            return ax

        positions = [item["position"] for item in combined_measurements]
        diameters = [item["diameter"] for item in combined_measurements]

        label_to_use = plot_label if plot_label else self.flute_data.flute_model

        ax.plot(positions, [d / 2.0 for d in diameters],
                color=flute_color if flute_color else BASE_COLORS[0],
                linestyle=flute_style if flute_style else LINESTYLES[0],
                linewidth=2, label=label_to_use)
        ax.plot(positions, [-d / 2.0 for d in diameters],
                color=flute_color if flute_color else BASE_COLORS[0],
                linestyle=flute_style if flute_style else LINESTYLES[0],
                linewidth=2)

        # TODO: Ploteo de agujeros en posiciones absolutas.
        # Ver plot_top_view_instrument_geometry para una implementación de esto.

        ax.set_xlabel("Posición (mm)")
        ax.set_ylabel("Radio (mm)")
        ax.set_aspect('equal', adjustable='datalim')
        if label_to_use: ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.7)
        title_str = "Vista 2D de la Flauta"
        if self.flute_data.flute_model and self.flute_data.flute_model != label_to_use:
            title_str += f" ({self.flute_data.flute_model})"
        ax.set_title(title_str)
        return ax

    def plot_instrument_geometry(self, note: str = "D", ax: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure
            ax.clear()

        try:
            if note not in self.flute_data.acoustic_analysis or not self.flute_data.acoustic_analysis[note]:
                msg = f"Análisis para nota '{note}' no disponible"
                logger.warning(f"{msg} en {self.flute_data.flute_model}.")
                ax.text(0.5,0.5, msg, ha='center', transform=ax.transAxes)
                ax.set_title(f"Geometría del Instrumento ({note}) - {self.flute_data.flute_model}")
                return ax

            acoustic_analysis_obj = self.flute_data.acoustic_analysis[note]
            if not isinstance(acoustic_analysis_obj, ImpedanceComputation):
                 msg = f"Datos de análisis inválidos para nota '{note}'"
                 logger.warning(f"{msg} en {self.flute_data.flute_model}.")
                 ax.text(0.5,0.5, msg, ha='center', transform=ax.transAxes)
                 ax.set_title(f"Geometría del Instrumento ({note}) - {self.flute_data.flute_model}")
                 return ax

            acoustic_analysis_obj.plot_instrument_geometry(ax=ax)

            ax.set_title(f"Geometría (Openwind) para {note} - {self.flute_data.flute_model}", fontsize=10)
            ax.set_xlabel("Posición (m)")
            ax.set_ylabel("Radio (m)")
            ax.grid(True, linestyle=':', alpha=0.7)
            return ax
        except Exception as e:
            logger.error(f"Error graficando geometría del instrumento para nota {note} en {self.flute_data.flute_model}: {e}")
            ax.text(0.5,0.5, f"Error graficando geometría para '{note}'", ha='center', transform=ax.transAxes, color='red')
            ax.set_title(f"Geometría del Instrumento ({note}) - Error", fontsize=10)
            return ax
        
    @staticmethod
    def _plot_shape_static(shape_data: Tuple[np.ndarray, np.ndarray], ax: plt.Axes, mmeter_conversion: float, **kwargs: Any) -> None:
        """ Dibuja una forma de tubo (x, r). Espera datos en metros, convierte a mm si mmeter_conversion = M_TO_MM_FACTOR. """
        x_m, r_m = shape_data

        # Convertir a la unidad deseada para el gráfico (ej. mm)
        x_plot = x_m * mmeter_conversion
        r_plot = r_m * mmeter_conversion

        # Añadir NaN para separar las dos mitades del perfil
        radius_to_plot = np.concatenate([r_plot, [np.nan], np.flip(-r_plot)])
        position_to_plot = np.concatenate([x_plot, [np.nan], np.flip(x_plot)])

        ax.plot(position_to_plot, radius_to_plot, **kwargs)

    @staticmethod
    def _plot_holes_static(
        holes_info: List[Dict[str, Any]], # Lista de diccionarios, cada uno con 'label', 'position_m', 'radius_m', 'is_open'
        ax: plt.Axes,
        mmeter_conversion: float,
        default_color: str = 'black',
        **kwargs: Any) -> None:
        try:
            theta = np.linspace(0, 2 * np.pi, 50)
            for hole_detail in holes_info:
                pos_m = hole_detail['position_m']
                rad_m = hole_detail['radius_m']
                is_open = hole_detail.get('is_open', True)
                # label = hole_detail.get('label', '') # No usado para el ploteo actual

                # Convertir a unidades de ploteo
                x_center_plot = pos_m * mmeter_conversion
                rad_plot = rad_m * mmeter_conversion

                x_circle_plot = x_center_plot + rad_plot * np.cos(theta)
                y_circle_plot = rad_plot * np.sin(theta) # Asume que el tubo está centrado en y=0

                plot_kwargs = kwargs.copy()
                current_color = plot_kwargs.pop('color', default_color) # Sacar color si está, usar default_color si no

                if is_open:
                    ax.plot(x_circle_plot, y_circle_plot, color=current_color, solid_capstyle='round', **plot_kwargs)
                else:
                    ax.fill(x_circle_plot, y_circle_plot, color=current_color, **plot_kwargs)
        except Exception as e:
            logger.error(f"Error graficando agujeros (estático): {e}")


    def plot_top_view_instrument_geometry(self, note: str = "D", ax: Optional[plt.Axes] = None) -> Optional[plt.Axes]:
        fig: plt.Figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 3))
        else:
            fig = ax.figure
            ax.clear()

        ax.set_aspect('equal', adjustable='datalim')

        try:
            if note not in self.flute_data.acoustic_analysis or not self.flute_data.acoustic_analysis[note]:
                msg = f"Análisis para nota '{note}' no disponible"
                logger.error(f"{msg} en {self.flute_data.flute_model}")
                ax.text(0.5,0.5, msg, ha='center', transform=ax.transAxes)
            elif not isinstance(self.flute_data.acoustic_analysis[note], ImpedanceComputation):
                msg = f"Datos de análisis inválidos para nota '{note}'"
                logger.error(f"{msg} en {self.flute_data.flute_model}")
                ax.text(0.5,0.5, msg, ha='center', transform=ax.transAxes)
            else:
                analysis_obj = self.flute_data.acoustic_analysis[note]
                instrument_geometry = analysis_obj.get_instrument_geometry()
                if not instrument_geometry:
                    msg = f"No hay datos de geometría para nota '{note}'"
                    logger.error(f"{msg} en {self.flute_data.flute_model}.")
                    ax.text(0.5,0.5, msg, ha='center', transform=ax.transAxes)
                else:
                    # Dibujar el perfil del tubo principal usando combined_measurements
                    combined_measurements = self.flute_data.combined_measurements
                    if combined_measurements:
                        try:
                            positions_mm = np.array([item["position"] for item in combined_measurements])
                            diameters_mm = np.array([item["diameter"] for item in combined_measurements])
                            radii_mm = diameters_mm / 2.0

                            ax.plot(positions_mm, radii_mm, color='black', linestyle='-', linewidth=1)
                            ax.plot(positions_mm, -radii_mm, color='black', linestyle='-', linewidth=1)
                            logger.debug(f"Dibujado Main Bore usando combined_measurements para {self.flute_data.flute_model} en vista superior.")
                        except Exception as e_plot_bore:
                            logger.error(f"Error al dibujar el tubo principal usando combined_measurements para {self.flute_data.flute_model}: {e_plot_bore}")
                            ax.text(0.5, 0.5, "Error al dibujar tubo", ha='center', va='center', transform=ax.transAxes, color='red')
                    else:
                        logger.warning(f"No hay mediciones combinadas para {self.flute_data.flute_model} para dibujar el tubo en vista superior.")
                        ax.text(0.5, 0.5, "Error: Geometría del tubo no disponible", ha='center', va='center', transform=ax.transAxes)

                    try:
                        # Plot Holes (positions are in meters from instrument_geometry.holes)
                        holes_details_for_plot = []
                        fingering = instrument_geometry.fingering_chart.fingering_of(note)
                        for hole_obj in instrument_geometry.holes:
                            pos_m = hole_obj.position.get_value()
                            rad_m = hole_obj.shape.get_radius_at(0) if hasattr(hole_obj.shape, 'get_radius_at') else 0.003 # Default radius if not found
                            is_open = fingering.is_side_comp_open(hole_obj.label)
                            holes_details_for_plot.append({
                                'label': hole_obj.label, 'position_m': pos_m, 'radius_m': rad_m, 'is_open': is_open
                            })
                        FluteOperations._plot_holes_static(holes_details_for_plot, ax, M_TO_MM_FACTOR, default_color='dimgray', linewidth=0.5)
                    except Exception as e_holes:
                        logger.error(f"Error al dibujar los agujeros para {self.flute_data.flute_model}, nota {note}: {e_holes}")
                        # No necesariamente mostrar un error en el gráfico por esto, ya que el tubo principal podría estar dibujado.

            ax.set_xlabel("Posición (mm)")
            ax.set_ylabel("Radio (mm)")
            ax.set_title(f"Vista Superior - Geometría para Nota '{note}' - {self.flute_data.flute_model}", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)
            return ax
        except Exception as e:
            logger.exception(f"Error generando vista superior de geometría para {self.flute_data.flute_model}, nota {note}: {e}")
            ax.text(0.5,0.5, f"Error generando vista para '{note}'", ha='center', transform=ax.transAxes, color='red')
            ax.set_title(f"Vista Superior ({note}) - Error", fontsize=10)
            return ax

    # --- Métodos Estáticos para Gráficos de Resumen (Comparativos) ---
    @staticmethod
    def plot_individual_admittance_analysis(
         acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
         combined_measurements_list: List[Tuple[List[Dict[str, float]], str]], # Lista de (combined_measurements, flute_name)
         note: str,
         fig_to_use: Optional[plt.Figure] = None,
         base_colors: List[str] = BASE_COLORS,
         linestyles: List[str] = LINESTYLES
     ) -> plt.Figure :

     fig: plt.Figure
     axes: np.ndarray

     if fig_to_use is not None:
         fig = fig_to_use
         fig.clear()
         axes_array = fig.subplots(4, 1, gridspec_kw={'height_ratios': [2, 1, 1, 1]})
         if not isinstance(axes_array, np.ndarray): axes = np.array([axes_array])
         else: axes = axes_array
     else:
         fig, axes_array = plt.subplots(4, 1, figsize=(12,18), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
         if not isinstance(axes_array, np.ndarray): axes = np.array([axes_array])
         else: axes = axes_array

     if not isinstance(axes, np.ndarray) or axes.ndim == 0 or axes.size < 4:
         logger.error("No se pudieron crear o obtener los ejes para plot_individual_admittance_analysis.")
         fig_fallback, ax_fallback = plt.subplots(); ax_fallback.text(0.5,0.5, "Error subplots")
         return fig_fallback

     ax_admittance, ax_pressure, ax_geometry, ax_flow = axes.flatten()
     
     # Limpiar ejes explícitamente si se reutiliza la figura
     if fig_to_use is not None:
         for ax_item in [ax_admittance, ax_pressure, ax_geometry, ax_flow]:
             ax_item.clear()

     legend_handles_adm, legend_handles_pres, legend_handles_flow = [], [], []

     for index, ((analysis_dict, flute_name_aa), (measurements_data, flute_name_cm)) in enumerate(zip(acoustic_analysis_list, combined_measurements_list)):
         # ... (lógica para plotear admitancia, presión, flujo como antes) ...
         style_idx = index % len(linestyles)
         color_idx = index % len(base_colors)
         linestyle = linestyles[style_idx]
         color = base_colors[color_idx]

         analysis_obj = analysis_dict.get(note)
         # Asegurarse que estamos usando el flute_name correcto y los datos correctos
         if flute_name_aa != flute_name_cm:
             logger.warning(f"Desajuste de nombres de flauta entre analysis_list ({flute_name_aa}) y measurements_list ({flute_name_cm}). Usando {flute_name_aa}.")
         flute_name = flute_name_aa # Usar el nombre de la lista de análisis como principal

         if not isinstance(analysis_obj, ImpedanceComputation):
             logger.debug(f"Análisis para nota '{note}' no disponible o inválido para {flute_name}.")
             continue

         frequencies = analysis_obj.frequencies
         impedance = analysis_obj.impedance
         valid_impedance = np.where(np.abs(impedance) < 1e-12, 1e-12, impedance)
         admittance_db = 20 * np.log10(np.abs(1.0 / valid_impedance))

         # Admittance Plot
         line_adm, = ax_admittance.plot(frequencies, admittance_db, linestyle=linestyle, color=color, label=flute_name, alpha=0.8)
         if not any(lh.get_label() == flute_name for lh in legend_handles_adm):
             legend_handles_adm.append(line_adm)

         antires_freqs = list(analysis_obj.antiresonance_frequencies())
         current_ymin_adm, current_ymax_adm = ax_admittance.get_ylim() if ax_admittance.has_data() else (np.min(admittance_db)-5 if admittance_db.size > 0 else -60, np.max(admittance_db)+5 if admittance_db.size > 0 else 0)
         ax_admittance.set_ylim(min(current_ymin_adm, np.min(admittance_db)-5 if admittance_db.size > 0 else -60),
                                max(current_ymax_adm, np.max(admittance_db)+5 if admittance_db.size > 0 else 0))
         ymin_adm, ymax_adm = ax_admittance.get_ylim()

         for i_ar, f_ar in enumerate(antires_freqs[:3]):
             ax_admittance.vlines(f_ar, ymin_adm, ymax_adm, color=color, linestyle=':', alpha=0.6)
             if i_ar < 2 :
                 ax_admittance.text(f_ar, ymin_adm + (ymax_adm - ymin_adm) * (0.95 - index*0.08), f"{f_ar:.0f}",
                                 rotation=90, color=color, fontsize=7, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5, pad=0.1, edgecolor='none'))

         # Pressure and Flow Plots
         x_coords, pressure_modes, flow_modes = analysis_obj.get_pressure_flow()
         pressure_abs = np.abs(pressure_modes.T)
         flow_abs = np.abs(flow_modes.T)

         # Iterar para los dos primeros armónicos (antirresonancias)
         for i_mode, f_mode in enumerate(antires_freqs[:2]): # Tomar hasta 2 antirresonancias
             if pressure_abs.shape[1] > 0 and flow_abs.shape[1] > 0:
                 idx_f_mode = np.argmin(np.abs(frequencies - f_mode))
                 if idx_f_mode < pressure_abs.shape[1]: # Comprobar que el índice es válido
                     # Usar un estilo de línea ligeramente diferente para el segundo armónico si es la misma flauta
                     mode_linestyle = linestyle if i_mode == 0 else '--' 
                     mode_alpha = 0.8 if i_mode == 0 else 0.6

                     line_pres, = ax_pressure.plot(x_coords, pressure_abs[:, idx_f_mode], 
                                                  linestyle=mode_linestyle, color=color,
                                                  label=f"{flute_name} (AR{i_mode+1}: {f_mode:.0f}Hz)", alpha=mode_alpha)
                     if not any(lh.get_label() == line_pres.get_label() for lh in legend_handles_pres):
                         legend_handles_pres.append(line_pres)

                     line_flow, = ax_flow.plot(x_coords, flow_abs[:, idx_f_mode], 
                                                  linestyle=mode_linestyle, color=color,
                                                  label=f"{flute_name} (AR{i_mode+1}: {f_mode:.0f}Hz)", alpha=mode_alpha)
                     if not any(lh.get_label() == line_flow.get_label() for lh in legend_handles_flow):
                         legend_handles_flow.append(line_flow)
         else:
             logger.debug(f"No hay frecuencias antiresonantes o datos de modo para {flute_name}, nota {note}.")


         # --- SECCIÓN MODIFICADA PARA EL PANEL DE GEOMETRÍA ---
         if ax_geometry:
             try:
                 class MinimalFluteDataForTopView:
                     def __init__(self, acoustic_analysis_data_for_note, model_name, combined_measurements_data):
                         self.acoustic_analysis = acoustic_analysis_data_for_note
                         self.flute_model = model_name
                         self.combined_measurements = combined_measurements_data
                         self.data = {"Flute Model": model_name}

                 # Encontrar los combined_measurements para la flauta actual
                 current_flute_measurements = []
                 for cm_data, cm_name in combined_measurements_list:
                     if cm_name == flute_name:
                         current_flute_measurements = cm_data; break

                 temp_flute_data_for_top_view = MinimalFluteDataForTopView({note: analysis_obj}, flute_name, current_flute_measurements)
                 temp_flute_ops_for_top_view = FluteOperations(temp_flute_data_for_top_view)

                 # Llamar a plot_top_view_instrument_geometry
                 # ax_geometry.clear() # plot_top_view_instrument_geometry ya debería limpiar su eje si se le pasa.
                 returned_ax = temp_flute_ops_for_top_view.plot_top_view_instrument_geometry(note=note, ax=ax_geometry)

                 if returned_ax is None:
                     logger.error(f"Plotting top view geometry falló y devolvió None para {flute_name}, nota {note}")
                     # ax_geometry ya podría tener un mensaje de error de la función de ploteo si falló internamente
                 elif returned_ax is not ax_geometry and returned_ax is not None : # type: ignore
                      # Esto no debería suceder si ax_geometry se pasa y se usa correctamente.
                     logger.warning("plot_top_view_instrument_geometry podría haber creado un nuevo eje inesperadamente. Cerrando figura extra.")
                     plt.close(returned_ax.figure) # type: ignore

             except Exception as e_geom_top_view:
                 logger.error(f"Error al graficar la vista superior de la geometría para {flute_name}, nota {note}: {e_geom_top_view}")
                 if ax_geometry: # Solo si el eje existe
                     ax_geometry.clear() 
                     ax_geometry.text(0.5,0.5, f"Error geom. sup. {flute_name}", ha='center', va='center', transform=ax_geometry.transAxes, color='red')
         # --- FIN DE LA SECCIÓN MODIFICADA ---

     # Configuración de títulos y etiquetas para los ejes (como estaba antes)
     if ax_admittance:
         ax_admittance.set_title(f"Admitancia para {note}", fontsize=10); ax_admittance.set_xlabel("Frecuencia (Hz)")
         ax_admittance.set_ylabel("Admitancia (dB)"); ax_admittance.legend(handles=legend_handles_adm, loc='best', fontsize=8); ax_admittance.grid(True, linestyle=':', alpha=0.7)
     if ax_pressure:
         ax_pressure.set_title(f"Presión vs Posición ({note})", fontsize=10); ax_pressure.set_xlabel("Posición (m)")
         ax_pressure.set_ylabel("Presión (Pa)"); ax_pressure.legend(handles=legend_handles_pres, loc='best', fontsize=8); ax_pressure.grid(True, linestyle=':', alpha=0.7)
     if ax_geometry: # El título lo pone plot_top_view_instrument_geometry
         ax_geometry.set_title(f"Geometría (Vista Sup.) para {note} ({flute_name})", fontsize=10); ax_geometry.set_xlabel("Posición (mm)")
         ax_geometry.set_ylabel("Radio (mm)"); ax_geometry.grid(True, linestyle=':', alpha=0.7)
     if ax_flow:
         ax_flow.set_title(f"Flujo vs Posición ({note})", fontsize=10); ax_flow.set_xlabel("Posición (m)")
         ax_flow.set_ylabel("Flujo (m³/s)"); ax_flow.legend(handles=legend_handles_flow, loc='best', fontsize=8); ax_flow.grid(True, linestyle=':', alpha=0.7)

     try:
         fig.tight_layout(rect=[0,0,1,0.97])
     except Exception as e_layout:
         logger.debug(f"Error en tight_layout para individual_admittance_analysis: {e_layout}")
     return fig

    @staticmethod
    def plot_combined_admittance(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            ax: Optional[plt.Axes] = None,
            base_colors: List[str] = BASE_COLORS,
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:

        fig: plt.Figure
        if ax is None: fig, ax = plt.subplots(figsize=(14, 8))
        else: fig = ax.figure; ax.clear()

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = base_colors[index % len(base_colors)]

            line_plotted_for_legend = False
            for note, analysis_obj in analysis_dict.items():
                if isinstance(analysis_obj, ImpedanceComputation):
                    frequencies = analysis_obj.frequencies
                    impedance = analysis_obj.impedance
                    valid_impedance = np.where(np.abs(impedance) < 1e-12, 1e-12, impedance)
                    admittance_db = 20 * np.log10(np.abs(1.0 / valid_impedance))

                    current_label = flute_name if not line_plotted_for_legend else "_nolegend_"
                    ax.plot(frequencies, admittance_db, linestyle=linestyle, color=color, label=current_label, alpha=0.6)
                    if not line_plotted_for_legend: line_plotted_for_legend = True

        handles, labels = ax.get_legend_handles_labels()
        unique_handles_labels = dict(zip(labels, handles))
        ax.legend(unique_handles_labels.values(), unique_handles_labels.keys(), loc='best', fontsize=9)

        ax.set_title("Admitancia Combinada (Todas las Notas, Superpuestas por Flauta)", fontsize=10)
        ax.set_xlabel("Frecuencia (Hz)"); ax.set_ylabel("Admitancia (dB)"); ax.grid(True, linestyle=':', alpha=0.7)
        return fig

    @staticmethod
    def plot_summary_antiresonances(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            notes_ordered: List[str],
            ax: Optional[plt.Axes] = None,
            base_colors: List[str] = BASE_COLORS
        ) -> plt.Figure:

        fig: plt.Figure
        if ax is None: fig, ax = plt.subplots(figsize=(14, 8))
        else: fig = ax.figure; ax.clear()

        num_flutes = len(acoustic_analysis_list)
        total_width_for_note = 0.7
        offsets = np.linspace(-total_width_for_note / 2, total_width_for_note / 2, num_flutes if num_flutes > 0 else 1) if num_flutes > 1 else [0]

        legend_handles = []

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            color = base_colors[index % len(base_colors)]

            for note_idx, note in enumerate(notes_ordered):
                analysis_obj = analysis_dict.get(note)
                if isinstance(analysis_obj, ImpedanceComputation):
                    antires_freqs = list(analysis_obj.antiresonance_frequencies())
                    if antires_freqs:
                        for i_ar, f_ar in enumerate(antires_freqs[:2]): # Max 2 ARs
                            x_pos = note_idx + offsets[index]
                            ax.plot(x_pos, f_ar, "o", color=color, markersize=6, alpha=0.7)
                            if i_ar < 2:
                                 ax.text(x_pos, f_ar + (10 * (-1)**i_ar), f"{f_ar:.0f}", fontsize=7,
                                         ha="center", va="bottom" if i_ar % 2 == 0 else "top", color=color,
                                         bbox=dict(facecolor='white', alpha=0.3, pad=0.1, edgecolor='none'))
            # Add to legend once per flute
            if not any(lh.get_label() == flute_name for lh in legend_handles):
                legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, linestyle='None', label=flute_name, markersize=6))

        ax.set_xticks(range(len(notes_ordered)))
        ax.set_xticklabels(notes_ordered, rotation=45, ha="right")
        ax.legend(handles=legend_handles, loc='best', fontsize=9)
        ax.set_title("Frecuencias Antiresonantes (Primeras 2) vs. Nota", fontsize=10)
        ax.set_xlabel("Nota"); ax.set_ylabel("Frecuencia (Hz)"); ax.grid(True, axis='y', linestyle=':', alpha=0.7)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_summary_cents_differences(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            notes_ordered: List[str],
            ax: Optional[plt.Axes] = None,
            base_colors: List[str] = BASE_COLORS,
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:

        fig: plt.Figure
        if ax is None: fig, ax = plt.subplots(figsize=(14, 8))
        else: fig = ax.figure; ax.clear()

        num_flutes = len(acoustic_analysis_list)
        offset_per_flute = 0.12
        base_x_positions = np.arange(len(notes_ordered))

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = base_colors[index % len(base_colors)]
            cents_diffs = []

            current_x_offset = (index - (num_flutes - 1) / 2.0) * offset_per_flute

            for note in notes_ordered:
                note_cents = np.nan
                analysis_obj = analysis_dict.get(note)
                if isinstance(analysis_obj, ImpedanceComputation):
                    antires_freqs = list(analysis_obj.antiresonance_frequencies())
                    if len(antires_freqs) >= 2:
                        f1, f2 = antires_freqs[0], antires_freqs[1]
                        if f1 > 0 and f2 > 0: note_cents = 1200 * np.log2(f2 / (2.0 * f1))
                cents_diffs.append(note_cents)

            ax.plot(base_x_positions + current_x_offset, cents_diffs, marker="o", linestyle=linestyle, color=color, label=flute_name, markersize=5, alpha=0.8)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', fontsize=9)
        ax.set_xticks(base_x_positions)
        ax.set_xticklabels(notes_ordered, rotation=45, ha="right")
        ax.axhline(0, color='grey', linestyle='--', lw=0.8)
        ax.set_title("Inharmonicidad (Cents: Pico 2 vs 2 * Pico 1)", fontsize=10)
        ax.set_xlabel("Nota"); ax.set_ylabel("Diferencia (cents)"); ax.grid(True, linestyle=':', alpha=0.7)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_moc_summary(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            finger_frequencies_map: Dict[str, Dict[str, float]],
            notes_ordered: List[str],
            ax: Optional[plt.Axes] = None,
            base_colors: List[str] = BASE_COLORS,
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:

        fig: plt.Figure
        if ax is None: fig, ax = plt.subplots(figsize=(12, 7))
        else: fig = ax.figure; ax.clear()

        num_flutes = len(acoustic_analysis_list)
        offset_per_flute = 0.12
        base_x_positions = np.arange(len(notes_ordered))

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            linestyle = linestyles[index % len(linestyles)]
            color = base_colors[index % len(base_colors)]
            moc_vals = []
            current_finger_freqs = finger_frequencies_map.get(flute_name, {})
            current_x_offset = (index - (num_flutes - 1) / 2.0) * offset_per_flute


            for note in notes_ordered:
                moc = np.nan
                f_play = current_finger_freqs.get(note)
                analysis_obj = analysis_dict.get(note)
                if isinstance(analysis_obj, ImpedanceComputation) and f_play is not None and f_play > 0 :
                    antires = list(analysis_obj.antiresonance_frequencies())
                    if len(antires) >= 2:
                        f0, f1 = antires[0], antires[1]
                        # Ensure f0, f1, f_play are positive and denominators are not zero
                        if f0 > 0 and f1 > 0 and f_play > 0 and f0 != f_play and (2.0 * f_play) != 0 and f1 != (2.0*f_play) :
                            num_term = (1.0 / f1) - (1.0 / (2.0 * f_play))
                            den_term = (1.0 / f0) - (1.0 / f_play)
                            if abs(den_term) > 1e-9: # Avoid division by zero
                                moc = num_term / den_term
                moc_vals.append(moc)

            ax.plot(base_x_positions + current_x_offset, moc_vals, marker="o", linestyle=linestyle, color=color, label=flute_name, markersize=5, alpha=0.8)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='best', fontsize=9)
        ax.set_xticks(base_x_positions)
        ax.set_xticklabels(notes_ordered, rotation=45, ha="right")
        ax.set_xlabel("Nota"); ax.set_ylabel("MOC (ratio)"); ax.set_title("Resumen de MOC por Nota", fontsize=10); ax.grid(True, linestyle=':', alpha=0.7)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_bi_espe_summary(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            finger_frequencies_map: Dict[str, Dict[str, float]],
            notes_ordered: List[str],
            ax: Optional[plt.Axes] = None,
            base_colors: List[str] = BASE_COLORS
        ) -> plt.Figure:

        fig: plt.Figure
        if ax is None: fig, ax = plt.subplots(figsize=(12, 7))
        else: fig = ax.figure; ax.clear()

        num_flutes = len(acoustic_analysis_list)
        base_x_positions = np.arange(len(notes_ordered))
        # Width for each flute's BI/ESPE pair, and spacing
        total_width_per_flute_group = 0.3
        inter_flute_spacing_factor = 0.15 # Factor of total_width_per_flute_group


        def get_speed_of_sound(temp_celsius=20.0): return 331.3 * np.sqrt(1 + temp_celsius / 273.15)
        speed_of_sound_ref = get_speed_of_sound(20.0)

        legend_items = {} # To store unique legend entries {label: handle}

        for idx, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            color = base_colors[idx % len(base_colors)]
            bi_vals, espe_vals = [], []
            current_finger_freqs = finger_frequencies_map.get(flute_name, {})

            for note in notes_ordered:
                bi, espe = np.nan, np.nan
                f_play_I = current_finger_freqs.get(note)
                analysis_obj = analysis_dict.get(note)

                if isinstance(analysis_obj, ImpedanceComputation) and f_play_I is not None and f_play_I > 0:
                    antires = list(analysis_obj.antiresonance_frequencies())
                    if len(antires) >= 2:
                        f0, f1 = antires[0], antires[1]
                        f_play_II = 2.0 * f_play_I

                        if f0 > 0: bi = 1200.0 * np.log2(f_play_I / f0)

                        delta_l_I = (speed_of_sound_ref / 2.0) * ((1.0 / f_play_I) - (1.0 / f0)) if f0 > 0 else 0.0
                        delta_l_II = speed_of_sound_ref * ((1.0 / f_play_II) - (1.0 / f1)) if f1 > 0 and f_play_II > 0 else 0.0
                        delta_delta_l = delta_l_II - delta_l_I
                        L_eff_I = (speed_of_sound_ref / (2.0 * f_play_I))

                        if L_eff_I > 0 and (L_eff_I + delta_delta_l) > 1e-9: # Avoid log of zero/negative and division by zero
                            espe = 1200.0 * np.log2(L_eff_I / (L_eff_I + delta_delta_l))
                bi_vals.append(bi); espe_vals.append(espe)

            # Calculate offset for this flute's group
            group_center_offset = (idx - (num_flutes - 1) / 2.0) * (total_width_per_flute_group * (1 + inter_flute_spacing_factor))
            
            # Plot BI
            line_bi, = ax.plot(base_x_positions + group_center_offset - total_width_per_flute_group/4, bi_vals,
                    linestyle='-', color=color, marker='o', markersize=5, alpha=0.8)
            # Plot ESPE
            line_espe, = ax.plot(base_x_positions + group_center_offset + total_width_per_flute_group/4, espe_vals,
                    linestyle='--', dashes=(4,2), color=color, marker='x', markersize=5, alpha=0.8) # Dashes for better distinction

            # Add to legend items only once per flute
            if f"{flute_name} - $B_I$" not in legend_items:
                legend_items[f"{flute_name} - $B_I$"] = line_bi
            if f"{flute_name} - ESPE" not in legend_items:
                legend_items[f"{flute_name} - ESPE"] = line_espe
        
        if legend_items:
            ax.legend(legend_items.values(), legend_items.keys(), fontsize=8, loc='best', ncol=max(1, num_flutes // 2))


        ax.set_xticks(base_x_positions)
        ax.set_xticklabels(notes_ordered, rotation=45, ha="right")
        ax.axhline(0, color='grey', linestyle='--', lw=0.8) # Linea en 0 cents
        ax.set_title("$B_I$ y ESPE a Través de las Notas", fontsize=10); ax.set_xlabel("Nota"); ax.set_ylabel("Cents"); ax.grid(True, linestyle=':', alpha=0.7)
        fig.tight_layout()
        return fig

    @staticmethod
    def generate_summary_pdf(
            pdf_filename: str,
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            finger_frequencies_map: Dict[str, Dict[str, float]],
            notes_ordered: List[str],
        ) -> str:
        with PdfPages(pdf_filename) as pdf:
            logger.info(f"Generando gráfico MOC para PDF: {pdf_filename}")
            fig_moc = FluteOperations.plot_moc_summary(acoustic_analysis_list, finger_frequencies_map, notes_ordered)
            pdf.savefig(fig_moc); plt.close(fig_moc)

            logger.info(f"Generando gráfico B_I/ESPE para PDF: {pdf_filename}")
            fig_bi_espe = FluteOperations.plot_bi_espe_summary(acoustic_analysis_list, finger_frequencies_map, notes_ordered)
            pdf.savefig(fig_bi_espe); plt.close(fig_bi_espe)

        logger.info(f"Reporte PDF de resumen guardado en: {pdf_filename}")
        return pdf_filename