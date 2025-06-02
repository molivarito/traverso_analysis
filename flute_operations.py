import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os # Mantener por si se usa en el futuro, aunque no en el código actual
from openwind import ImpedanceComputation, Player, InstrumentGeometry
import numpy as np
# from matplotlib.cm import tab10 # No usado directamente, se puede quitar si no se usa para generar colores dinámicamente aquí
import logging
from typing import Any, List, Tuple, Optional, Dict

# Importar constantes
from constants import (
    BASE_COLORS, LINESTYLES, FLUTE_PARTS_ORDER,
    M_TO_MM_FACTOR, MM_TO_M_FACTOR # MM_TO_M_FACTOR no usado aquí, pero M_TO_MM_FACTOR sí
)

logger = logging.getLogger(__name__)

class FluteOperations:
    def __init__(self, flute_data: Any) -> None:
        self.flute_data = flute_data
        # El color individual podría ser útil si esta clase se usa para una sola flauta.
        # Sin embargo, en la GUI y data_processing, los colores se asignan en el bucle.
        # self.flute_color: str = self._assign_color(flute_data.data.get("Flute Model", "Default"))

    # _assign_color no se usa si los colores se manejan externamente. Se puede quitar o mantener por si acaso.
    # @staticmethod
    # def _assign_color(flute_name: str) -> str:
    #     """
    #     Assigns a unique color based on the flute name.
    #     """
    #     index = hash(flute_name) % len(BASE_COLORS)
    #     return BASE_COLORS[index]

    def _calculate_adjusted_positions(self, part: str, current_position: float) -> Tuple[List[float], List[float]]:
        positions = [item["position"] for item in self.flute_data.data[part].get("measurements", [])]
        diameters = [item["diameter"] for item in self.flute_data.data[part].get("measurements", [])]
        adjusted_positions = [pos + current_position for pos in positions]
        return adjusted_positions, diameters

    def plot_individual_parts(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                              flute_color: Optional[str] = None) -> Tuple[plt.Figure, List[Any]]:
        # Usa FLUTE_PARTS_ORDER de constants
        # Usa LINESTYLES de constants

        if ax is None:
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            ax_list = axes.flatten() # Renombrar para evitar confusión con el argumento 'ax'
        elif isinstance(ax, plt.Axes): # Un solo eje, para una sola parte (menos común para este método)
            fig = ax.figure
            ax_list = [ax]
        else: # Lista de ejes
            fig = ax[0].figure
            ax_list = ax


        if len(ax_list) < len(FLUTE_PARTS_ORDER):
            logger.warning("No hay suficientes ejes para todas las partes de la flauta en plot_individual_parts.")
            # Podría lanzar un error o solo graficar en los ejes disponibles.
            # Por ahora, se truncará si ax_list es más corto.

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            if i >= len(ax_list): # No más ejes disponibles
                break
            current_ax = ax_list[i]
            
            # El '0' como current_position es para mostrar cada parte desde su propio origen local.
            adjusted_positions, diameters = self._calculate_adjusted_positions(part_name, 0)
            
            linestyle = LINESTYLES[i % len(LINESTYLES)]
            label = self.flute_data.data.get("Flute Model", "Unknown") if flute_names else None # El label se usa en la leyenda del subplot

            current_ax.plot(adjusted_positions, diameters, marker='o', linestyle=linestyle,
                       color=flute_color, markersize=4, label=label)

            hole_positions = self.flute_data.data[part_name].get("Holes position", [])
            hole_diameters = self.flute_data.data[part_name].get("Holes diameter", [])
            if hole_positions and hole_diameters:
                # Y-position para agujeros es arbitraria, ajustar si es necesario (ej. -max(diameters)/4)
                hole_plot_y_pos = 10 # o min(diameters) - 5 o algo relativo
                for pos, diam in zip(hole_positions, hole_diameters):
                    # El tamaño del marcador podría ser relativo al diámetro del agujero
                    current_ax.plot(pos, hole_plot_y_pos, color=flute_color, marker='o', markersize=max(diam * 0.5, 2))

            current_ax.set_xlabel("Posición (mm)")
            current_ax.set_ylabel("Diámetro (mm)")
            current_ax.set_title(f"{part_name.capitalize()} Geometría")
            current_ax.grid(True)
            if label: current_ax.legend(loc='best', fontsize=8)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if flute_names: # El título general de la figura
            fig.suptitle(", ".join(flute_names), fontsize=12) # Esto podría ser el nombre de la flauta actual

        return fig, ax_list


    def plot_all_parts_overlapping(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                                   flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> Any:
        """
        Grafica todas las partes de la flauta posicionadas secuencialmente (extremo a extremo),
        sin considerar la superposición real de mortajas/espigas.
        Para un perfil ensamblado realista, usar plot_combined_flute_data.
        """
        if ax is None:
            fig = plt.figure(figsize=(20, 12)) # Este tamaño es bastante grande
            ax = fig.add_subplot(111)

        current_position = 0.0
        plot_label = self.flute_data.data.get("Flute Model", "Unknown") if flute_names else None

        for i, part_name in enumerate(FLUTE_PARTS_ORDER):
            # El 'current_position' aquí es para mostrar las partes una después de la otra,
            # usando la longitud total de la parte anterior para el desplazamiento.
            # No usa la lógica de 'combine_measurements'.
            adjusted_positions, diameters = self._calculate_adjusted_positions(part_name, current_position)
            
            # Etiquetar solo la primera parte para la leyenda general de esta flauta
            label_for_part = plot_label if i == 0 else None
            
            ax.plot(adjusted_positions, diameters, marker='o', linestyle=flute_style,
                    color=flute_color, markersize=4, label=label_for_part)
      
            hole_positions = self.flute_data.data[part_name].get("Holes position", [])
            hole_diameters = self.flute_data.data[part_name].get("Holes diameter", [])
            if hole_positions and hole_diameters:
                hole_plot_y_pos = 10 # o min(diameters) - 5
                for pos, diam in zip(hole_positions, hole_diameters):
                    ax.plot(pos + current_position, hole_plot_y_pos, color=flute_color, marker='o', markersize=max(diam * 0.5, 2))

            total_length = self.flute_data.data[part_name].get("Total length", 0.0)
            current_position += total_length # Desplazar por la longitud total de la parte actual

        ax.set_xlabel("Posición Acumulada (mm)")
        ax.set_ylabel("Diámetro (mm)")
        if plot_label: ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title("Partes de la Flauta Desplegadas Secuencialmente" + (f" - {', '.join(flute_names)}" if flute_names else ""))
        return ax

    def plot_combined_flute_data(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                                 flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> Any:
        if ax is None:
            fig = plt.figure(figsize=(20, 12)) # Tamaño grande
            ax = fig.add_subplot(111)

        # Usa las mediciones combinadas de FluteData, que maneja la superposición
        combined_measurements = self.flute_data.combined_measurements # Ya calculado en FluteData.__init__
        positions = [item["position"] for item in combined_measurements]
        diameters = [item["diameter"] for item in combined_measurements]
        
        plot_label = self.flute_data.data.get("Flute Model", "Unknown") if flute_names else "Perfil Combinado"

        ax.plot(positions, diameters, label=plot_label,
                linestyle=flute_style, color=flute_color)

        # Para los agujeros, necesitamos sus posiciones absolutas en el instrumento ensamblado.
        # Esta información está implícita en los datos de `side_holes` de `compute_acoustic_analysis`.
        # O, recalcular aquí con la misma lógica de `compute_acoustic_analysis` para las posiciones de los agujeros.
        # Por ahora, se omite el ploteo de agujeros aquí para evitar duplicar/desincronizar esa lógica compleja.
        # Idealmente, FluteData podría pre-calcular las posiciones absolutas de los agujeros.
        # Si se grafica `instrument_geometry` o `top_view_instrument_geometry`, eso ya muestra los agujeros.
        
        # Ejemplo simplificado si se quisiera (requiere lógica robusta para pos_abs_agujero):
        # for part_name in FLUTE_PARTS_ORDER:
        #     for pos_rel_agujero, diam_agujero in zip(..., ...):
        #         pos_abs_agujero = self.flute_data.obtener_pos_abs_agujero(part_name, pos_rel_agujero) # Método hipotético
        #         ax.plot(pos_abs_agujero, 10, marker='o', color=flute_color, markersize=diam_agujero*0.5)


        ax.set_xlabel("Posición (mm)")
        ax.set_ylabel("Diámetro (mm)")
        ax.grid(True)
        if plot_label: ax.legend(loc='upper right')
        ax.set_title("Perfil de Flauta Combinado" + (f" - {', '.join(flute_names)}" if flute_names else ""))
        return ax

    def plot_flute_2d_view(self, ax: Optional[Any] = None, flute_names: Optional[List[str]] = None,
                           flute_color: Optional[str] = None, flute_style: Optional[str] = None) -> Any:
        if ax is None:
            fig = plt.figure(figsize=(15, 3))
            ax = fig.add_subplot(111)

        combined_measurements = self.flute_data.combined_measurements
        positions = [item["position"] for item in combined_measurements]
        diameters = [item["diameter"] for item in combined_measurements]
        
        plot_label = self.flute_data.data.get("Flute Model", "Unknown") if flute_names else "Vista 2D"

        ax.plot(positions, [d / 2 for d in diameters], color=flute_color, linestyle=flute_style,
                linewidth=2, label=plot_label)
        ax.plot(positions, [-d / 2 for d in diameters], color=flute_color, linestyle=flute_style, linewidth=2)

        # Similar a plot_combined_flute_data, el ploteo de agujeros aquí requiere sus posiciones absolutas.
        # Se omite por la misma razón de complejidad y para evitar desincronización.
        # La vista `plot_top_view_instrument_geometry` es más adecuada para esto.

        ax.set_xlabel("Posición (mm)")
        ax.set_ylabel("Radio (mm)")
        ax.set_aspect('equal', adjustable='datalim')
        if plot_label: ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title("Vista 2D de la Flauta" + (f" - {', '.join(flute_names)}" if flute_names else ""))
        return ax

    def plot_instrument_geometry(self, note: str = "D", ax: Optional[Any] = None) -> Optional[Any]:
        # Este método es específico de una instancia de FluteOperations (una flauta)
        try:
            if note not in self.flute_data.acoustic_analysis:
                logger.warning(f"Análisis acústico para la nota '{note}' no encontrado en {self.flute_data.data.get('Flute Model')}.")
                return None
            
            acoustic_analysis = self.flute_data.acoustic_analysis[note]
            if not isinstance(acoustic_analysis, ImpedanceComputation):
                 logger.warning(f"Análisis acústico para nota '{note}' no es del tipo esperado en {self.flute_data.data.get('Flute Model')}.")
                 return None

            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 6)) # Ajustar tamaño según necesidad
            
            # El método de ploteo de openwind
            acoustic_analysis.plot_instrument_geometry(ax=ax)
            
            ax.set_title(f"Geometría del Instrumento para {note} - {self.flute_data.data.get('Flute Model', '')}")
            # Labels X, Y son usualmente puestos por openwind, pero se pueden sobreescribir/mejorar si es necesario
            # ax.set_xlabel("Posición (m)") # Openwind usualmente usa metros
            # ax.set_ylabel("Radio (m)")
            ax.grid(True)
            return ax
        except Exception as e:
            logger.error(f"Error graficando geometría del instrumento para nota {note} en {self.flute_data.data.get('Flute Model', '')}: {e}")
            return None

    @staticmethod
    def plot_individual_admittance_analysis(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]], 
            note: str, 
            base_colors: List[str] = BASE_COLORS, 
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:
        """
        Crea un gráfico multi-panel mostrando admitancia, presión, geometría y flujo para una nota específica,
        comparando múltiples flautas.
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [2, 1, 1, 1]}) # Ajustar tamaño
        ax_admittance, ax_pressure, ax_geometry, ax_flow = axes

        # Leyendas para cada subplot
        legend_handles_adm, legend_handles_pres, legend_handles_flow = [], [], []

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            style_idx = index % len(linestyles)
            color_idx = index % len(base_colors)
            linestyle = linestyles[style_idx]
            color = base_colors[color_idx]
            
            if note in analysis_dict:
                analysis_obj = analysis_dict[note]
                if not isinstance(analysis_obj, ImpedanceComputation):
                    logger.warning(f"Análisis para {note} en {flute_name} no es ImpedanceComputation. Saltando.")
                    continue

                frequencies = analysis_obj.frequencies
                impedance = analysis_obj.impedance
                admittance = 1 / impedance # Cuidado con impedancia cero, aunque es improbable para np.array

                # Admitancia
                line_adm, = ax_admittance.plot(frequencies, 20 * np.log10(np.abs(admittance)),
                                   linestyle=linestyle, color=color, label=flute_name)
                if index == 0: legend_handles_adm.append(line_adm) # Añadir a leyenda solo una vez por flauta (o manejar duplicados)
                
                antiresonant_frequencies = list(analysis_obj.antiresonance_frequencies())
                ymin_adm, ymax_adm = ax_admittance.get_ylim() # Obtener límites después de graficar
                for i, f_ar in enumerate(antiresonant_frequencies[:5]): # Limitar a 5 para no saturar
                    ax_admittance.vlines(f_ar, ymin_adm, ymax_adm, color=color, linestyle=':', alpha=0.7)
                    if i < 2 : # Etiquetar solo las primeras 2 para claridad
                        ax_admittance.text(f_ar, ymin_adm + (ymax_adm - ymin_adm) * (0.9 - index*0.05) , f"{f_ar:.0f}Hz",
                                           rotation=90, color=color, fontsize=8, ha='right', va='top')

                # Presión y Flujo
                x_coords, pressure_modes, flow_modes = analysis_obj.get_pressure_flow()
                pressure_abs = np.abs(pressure_modes.T)
                flow_abs = np.abs(flow_modes.T)

                if antiresonant_frequencies:
                    idx_f1 = np.argmin(np.abs(frequencies - antiresonant_frequencies[0]))
                    
                    line_pres1, = ax_pressure.plot(x_coords, pressure_abs[:, idx_f1], linestyle=linestyle, color=color,
                                     label=f"{flute_name} ({antiresonant_frequencies[0]:.0f} Hz)")
                    line_flow1, = ax_flow.plot(x_coords, flow_abs[:, idx_f1], linestyle=linestyle, color=color,
                                label=f"{flute_name} ({antiresonant_frequencies[0]:.0f} Hz)")
                    if index == 0: 
                        legend_handles_pres.append(line_pres1)
                        legend_handles_flow.append(line_flow1)


                    if len(antiresonant_frequencies) > 1:
                        idx_f2 = np.argmin(np.abs(frequencies - antiresonant_frequencies[1]))
                        line_pres2, = ax_pressure.plot(x_coords, pressure_abs[:, idx_f2], linestyle='--', dashes=(5,5) if linestyle=='-' else linestyle, color=color, alpha=0.7, # Diferenciar estilo
                                         label=f"{flute_name} ({antiresonant_frequencies[1]:.0f} Hz)")
                        line_flow2, = ax_flow.plot(x_coords, flow_abs[:, idx_f2], linestyle='--', dashes=(5,5) if linestyle=='-' else linestyle, color=color, alpha=0.7,
                                    label=f"{flute_name} ({antiresonant_frequencies[1]:.0f} Hz)")
                        # Podríamos decidir no añadir estas segundas líneas a la leyenda principal para simplificarla.
                
                # Geometría (usando el método de openwind)
                try:
                    # Para la geometría, es mejor graficar cada una con un ligero offset vertical
                    # o usar alfa, pero InstrumentGeometry.plot_instrument_geometry no lo soporta directamente.
                    # Una alternativa es extraer los datos y graficarlos manualmente si se necesita control fino.
                    # Por ahora, se superpondrán.
                    # Se podría crear un nuevo eje para cada flauta si son pocas.
                    # O usar el método de FluteOperations para una sola flauta:
                    temp_flute_ops = FluteOperations(type('DummyFluteData', (object,), {'acoustic_analysis': analysis_dict, 'data': {'Flute Model': flute_name}})())
                    temp_flute_ops.plot_instrument_geometry(note=note, ax=ax_geometry)
                    # Esto es un hack. Sería mejor que plot_instrument_geometry fuera más flexible o
                    # que _plot_shape y _plot_holes fueran accesibles y reutilizables aquí.
                    # Re-implementación simplificada para el contexto actual:
                    # instrument_geometry_obj = analysis_obj.get_instrument_geometry()
                    # if instrument_geometry_obj:
                    #     for shape_geom in instrument_geometry_obj.main_bore_shapes:
                    #         FluteOperations._plot_shape_static(shape_geom, ax_geometry, mmeter=M_TO_MM_FACTOR, color=color, linewidth=1, alpha=0.7, label=flute_name if index == 0 else None) # Necesitaría _plot_shape_static
                    #     FluteOperations._plot_holes_static(instrument_geometry_obj.holes, ax_geometry, mmeter=M_TO_MM_FACTOR, note=note, acoustic_analysis_obj=analysis_obj, color=color, alpha=0.7) # Necesitaría _plot_holes_static
                    # ax_geometry.set_aspect('equal', adjustable='datalim') # Esto puede ser problemático con múltiples geometrías superpuestas

                except Exception as e_geom:
                    logger.error(f"Error graficando geometría para {flute_name}, nota {note}: {e_geom}")

        ax_admittance.set_title(f"Admitancia para {note}")
        ax_admittance.set_xlabel("Frecuencia (Hz)")
        ax_admittance.set_ylabel("Admitancia (dB)")
        ax_admittance.legend(handles=legend_handles_adm, loc='upper right', fontsize=8)
        ax_admittance.grid(True)

        ax_pressure.set_title(f"Presión vs Posición para {note} (primeras anti-resonancias)")
        ax_pressure.set_xlabel("Posición (m)")
        ax_pressure.set_ylabel("Presión (Pa)")
        ax_pressure.legend(handles=legend_handles_pres, loc='upper right', fontsize=8)
        ax_pressure.grid(True)

        ax_geometry.set_title(f"Geometría del Instrumento (Superpuesta) para {note}")
        ax_geometry.set_xlabel("Posición (m)") # Openwind usa metros
        ax_geometry.set_ylabel("Radio (m)")
        ax_geometry.grid(True)
        # La leyenda para la geometría es más compleja si se superponen varias.
        # Podría necesitarse un manejo especial si se usa _plot_shape_static/_plot_holes_static.

        ax_flow.set_title(f"Flujo vs Posición para {note} (primeras anti-resonancias)")
        ax_flow.set_xlabel("Posición (m)")
        ax_flow.set_ylabel("Flujo (m³/s)") # Verificar unidades de openwind
        ax_flow.legend(handles=legend_handles_flow, loc='upper right', fontsize=8)
        ax_flow.grid(True)

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_combined_admittance(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]],
            base_colors: List[str] = BASE_COLORS, 
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:
        fig = plt.figure(figsize=(14, 8)) # Ajustar tamaño
        ax = fig.add_subplot(111)
        legend_handles = []

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            style_idx = index % len(linestyles)
            color_idx = index % len(base_colors)
            linestyle = linestyles[style_idx]
            color = base_colors[color_idx]
            
            # Para la leyenda, solo tomar la primera línea de esta flauta
            line_plotted_for_legend = False
            for note, analysis_obj in analysis_dict.items():
                if isinstance(analysis_obj, ImpedanceComputation):
                    # El método .plot_admittance de openwind grafica directamente.
                    # Para control total, extraer datos y graficar con ax.plot.
                    # analysis_obj.plot_admittance(figure=fig, linestyle=linestyle, color=color) # Esto añade al fig/ax actual
                    
                    # Implementación manual para mejor control de leyenda:
                    frequencies = analysis_obj.frequencies
                    impedance = analysis_obj.impedance
                    admittance_db = 20 * np.log10(np.abs(1 / impedance))
                    line, = ax.plot(frequencies, admittance_db, linestyle=linestyle, color=color, label=f"{flute_name} - {note}" if not line_plotted_for_legend else "_nolegend_")
                    if not line_plotted_for_legend:
                        # Crear un handle solo para el nombre de la flauta
                        legend_handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, label=flute_name))
                        line_plotted_for_legend = True
                else:
                    logger.warning(f"Análisis para {note} en {flute_name} no es ImpedanceComputation.")
            
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
        ax.set_title("Admitancia Combinada para Todas las Notas (Superpuestas por Flauta)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Admitancia (dB)")
        ax.grid(True)
        return fig

    @staticmethod
    def plot_summary_antiresonances(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]], 
            notes_ordered: List[str],
            base_colors: List[str] = BASE_COLORS,
            # linestyles: List[str] = LINESTYLES # No se usa linestyle para scatter
        ) -> plt.Figure:
        fig = plt.figure(figsize=(14, 8)) # Ajustar tamaño
        ax = fig.add_subplot(111)
        
        num_flutes = len(acoustic_analysis_list)
        # Desplazamiento para separar puntos de diferentes flautas para la misma nota
        # El ancho total para los puntos de una nota será (num_flutes - 1) * offset
        # Queremos que esto sea < 1, e.g., 0.6. offset = 0.6 / (num_flutes -1) if num_flutes > 1 else 0
        offset_factor = 0.15 # Pequeño factor para desplazar puntos
        bar_width_total_for_note = 0.8 # Ancho total asignado a los puntos de una nota

        if num_flutes > 1:
            offsets = np.linspace(-bar_width_total_for_note / 2, bar_width_total_for_note / 2, num_flutes)
        else:
            offsets = [0]


        legend_handles = []

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            # style_idx = index % len(linestyles) # No usado para scatter
            color_idx = index % len(base_colors)
            # linestyle = linestyles[style_idx] # No usado
            color = base_colors[color_idx]
            
            current_flute_x_coords = []
            current_flute_y_freqs = []
            
            for note_idx, note in enumerate(notes_ordered):
                if note in analysis_dict:
                    analysis_obj = analysis_dict[note]
                    if isinstance(analysis_obj, ImpedanceComputation):
                        antires_freqs = list(analysis_obj.antiresonance_frequencies())
                        if antires_freqs:
                            # Graficar solo las primeras N (e.g., 3) anti-resonancias para claridad
                            for i_ar, f_ar in enumerate(antires_freqs[:3]):
                                x_pos = note_idx + offsets[index]
                                ax.plot(x_pos, f_ar, "o", color=color, markersize=5, alpha=0.8) # Scatter plot
                                if i_ar < 2: # Etiquetar las primeras 2
                                     ax.text(x_pos, f_ar, f"{f_ar:.0f}", fontsize=7, ha="center", va="bottom" if i_ar % 2 == 0 else "top", rotation=45, color=color)
                    else:
                        logger.warning(f"Análisis para {note} en {flute_name} no es ImpedanceComputation.")
            
            # Añadir a la leyenda
            legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, linestyle='None', label=flute_name))

        ax.set_xticks(range(len(notes_ordered)))
        ax.set_xticklabels(notes_ordered)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
        ax.set_title("Frecuencias Antiresonantes vs. Nota")
        ax.set_xlabel("Nota")
        ax.set_ylabel("Frecuencia (Hz)")
        ax.grid(True, axis='y', linestyle='--') # Grid solo en Y para claridad con los xticks
        return fig

    @staticmethod
    def plot_summary_cents_differences(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]], 
            notes_ordered: List[str],
            base_colors: List[str] = BASE_COLORS, 
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:
        fig = plt.figure(figsize=(14, 8)) # Ajustar tamaño
        ax = fig.add_subplot(111)
        legend_handles = []

        num_flutes = len(acoustic_analysis_list)
        offset_factor = 0.15
        if num_flutes > 1:
            offsets = np.linspace(-offset_factor * (num_flutes-1)/2, offset_factor * (num_flutes-1)/2, num_flutes)
        else:
            offsets = [0]

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            style_idx = index % len(linestyles)
            color_idx = index % len(base_colors)
            linestyle = linestyles[style_idx]
            color = base_colors[color_idx]
            
            cents_differences = []
            x_positions_for_plot = []

            for note_idx, note in enumerate(notes_ordered):
                note_cents_diff = np.nan # Valor por defecto si no se puede calcular
                if note in analysis_dict:
                    analysis_obj = analysis_dict[note]
                    if isinstance(analysis_obj, ImpedanceComputation):
                        antiresonant_frequencies = list(analysis_obj.antiresonance_frequencies())
                        if len(antiresonant_frequencies) >= 2:
                            f1, f2 = antiresonant_frequencies[0], antiresonant_frequencies[1]
                            if f1 > 0 and f2 > 0: # Asegurar frecuencias positivas
                                note_cents_diff = 1200 * np.log2(f2 / (2 * f1))
                    else:
                        logger.warning(f"Análisis para {note} en {flute_name} no es ImpedanceComputation.")
                cents_differences.append(note_cents_diff)
                x_positions_for_plot.append(note_idx + offsets[index])
            
            ax.plot(x_positions_for_plot, cents_differences, marker="o", linestyle=linestyle, color=color, label=flute_name if index == 0 else "_nolegend_")
            legend_handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, marker='o', label=flute_name))

        ax.set_xticks(range(len(notes_ordered)))
        ax.set_xticklabels(notes_ordered)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
        ax.set_title("Inharmonicidad (Diferencia en Cents: Pico 2 vs 2 * Pico 1)")
        ax.set_xlabel("Nota")
        ax.set_ylabel("Diferencia (cents)")
        ax.grid(True)
        return fig

    # --- Métodos de ayuda para graficar geometría ---
    # Estos métodos son casi copias de los de openwind. Si openwind los expone o permite más personalización,
    # sería mejor usarlos directamente. Hacerlos estáticos si no dependen de 'self'.
    @staticmethod
    def _plot_shape_static(shape: Any, ax: Any, mmeter: float, shift_x: float = 0, shift_y: float = 0, **kwargs: Any) -> None:
        """
        Dibuja una forma de tubo. (Adaptado para ser estático)
        """
        x, r = InstrumentGeometry._get_xr_shape(shape) # Esto es de openwind, podría no ser accesible directamente.
                                                      # Necesitaría una forma de obtener x,r de la 'shape'
                                                      # Si 'shape' es un objeto con métodos .get_x() .get_r(), usar eso.
                                                      # Asumamos que 'shape' es una tupla (x_coords, r_coords)
        if hasattr(shape, 'get_geometry_points'): # Para objetos de forma de openwind
             x_points, r_points = shape.get_geometry_points()
             x = np.array(x_points)
             r = np.array(r_points)
        else: # Fallback si es una estructura simple
            x, r = shape # Asumiendo que shape es (x_data, radius_data)
        
        radius_plot = np.append(r, np.nan) # NaN para romper la línea en el flip
        position_plot = np.append(x, np.nan) + shift_x
        
        ax.plot(np.concatenate([position_plot, np.flip(position_plot)]) * mmeter,
                (np.concatenate([radius_plot, np.flip(-radius_plot)]) + shift_y) * mmeter,
                **kwargs)

    @staticmethod
    def _plot_holes_static(holes: List[Any], ax: Any, mmeter: float, note: Optional[str] = None, 
                           acoustic_analysis_obj: Optional[ImpedanceComputation] = None, **kwargs: Any) -> None:
        """
        Dibuja los agujeros. (Adaptado para ser estático)
        """
        try:
            theta = np.linspace(0, 2 * np.pi, 50) # Menos puntos para el círculo
            for hole_obj in holes: # hole_obj es un objeto de openwind.Hole
                position_m = hole_obj.position.get_value()
                # El radio del agujero puede ser complejo (cónico, etc.). Tomar un promedio o el radio interno.
                # hole_obj.shape es la forma de la chimenea.
                # radius_m = np.mean(hole_obj.shape.get_radius_at(np.linspace(0, 1, 10)))
                # O el radio de entrada si es un cilindro simple:
                if hasattr(hole_obj.shape, 'radius'): # Para formas simples como Cylinder
                    radius_m = hole_obj.shape.radius.get_value()
                else: # Para formas más complejas, tomar el radio en la base de la chimenea
                    radius_m = hole_obj.shape.get_radius_at(0)


                x_circle_m = position_m + radius_m * np.cos(theta)
                y_circle_m = radius_m * np.sin(theta)
                
                is_open = True # Por defecto abierto si no hay info
                if note and acoustic_analysis_obj:
                    fingering = acoustic_analysis_obj.get_instrument_geometry().fingering_chart.fingering_of(note)
                    is_open = fingering.is_side_comp_open(hole_obj.label)
                
                plot_func = ax.plot if is_open else ax.fill
                # Kwargs pasados podrían incluir 'color'
                plot_kwargs = kwargs.copy()
                if not is_open and 'color' in plot_kwargs:
                    plot_kwargs.setdefault('edgecolor', plot_kwargs['color']) # Borde del mismo color que el relleno
                
                hole_plot = plot_func(x_circle_m * mmeter, y_circle_m * mmeter, **plot_kwargs)
                # if hole_plot and hasattr(hole_plot[0], "set_edgecolor") and not is_open:
                #     hole_plot[0].set_edgecolor(hole_plot[0].get_facecolor()) # type: ignore

        except Exception as e:
            logger.error(f"Error graficando agujeros (estático): {e}")


    def plot_top_view_instrument_geometry(self, note: str = "D") -> Optional[plt.Figure]:
        """
        Genera la vista superior de la geometría del instrumento con agujeros para una nota específica.
        Este método sigue siendo de instancia ya que usa self.flute_data.
        """
        try:
            fig, ax = plt.subplots(figsize=(15, 3)) # Tamaño estándar para esta vista
            ax.set_aspect('equal', adjustable='datalim')
            
            if note not in self.flute_data.acoustic_analysis:
                logger.error(f"Análisis para nota {note} no encontrado en {self.flute_data.data.get('Flute Model')}")
                plt.close(fig)
                return None
            
            analysis_obj = self.flute_data.acoustic_analysis[note]
            if not isinstance(analysis_obj, ImpedanceComputation):
                logger.error(f"Análisis para nota {note} no es ImpedanceComputation en {self.flute_data.data.get('Flute Model')}")
                plt.close(fig)
                return None

            instrument_geometry = analysis_obj.get_instrument_geometry()
            if not instrument_geometry:
                logger.error(f"No se pudo obtener la geometría del instrumento para {self.flute_data.data.get('Flute Model')}, nota {note}.")
                plt.close(fig)
                return None
            
            logger.info(f"Graficando forma del tubo principal para {self.flute_data.data.get('Flute Model')}, nota {note}...")
            # Graficar el cuerpo principal
            for shape_geom in instrument_geometry.main_bore_shapes:
                 # Necesitamos una manera de obtener los puntos x, r de shape_geom.
                 # Si plot_instrument_geometry de openwind es suficiente, se podría llamar y luego añadir los agujeros.
                 # O usar una reimplementación como _plot_shape_static.
                 # Ejemplo usando _plot_shape_static (requiere que shape_geom sea compatible o adaptado)
                 # FluteOperations._plot_shape_static(shape_geom, ax, M_TO_MM_FACTOR, color='black', linewidth=1)
                 # Por ahora, confiamos en que plot_instrument_geometry dibuja el cuerpo, luego añadimos agujeros.
                 # Esto es menos ideal si plot_instrument_geometry no toma un 'ax' o no se puede superponer.
                 # Mejor: llamar al método de instancia plot_instrument_geometry que ya maneja esto.
                 self.plot_instrument_geometry(note=note, ax=ax) # Esto dibuja el cuerpo.
                 # Ahora, plot_instrument_geometry puede haber puesto su propio título, etc.
                 # Quizás necesitemos una versión de plot_instrument_geometry que solo dibuje el cuerpo.


            # Graficar los agujeros usando el método estático
            FluteOperations._plot_holes_static(
                instrument_geometry.holes, 
                ax, 
                M_TO_MM_FACTOR, # Convertir metros (de openwind) a mm para el gráfico
                note=note, 
                acoustic_analysis_obj=analysis_obj,
                color=BASE_COLORS[0] # Usar un color base para los agujeros, o el color de la flauta
            )
            
            ax.set_xlabel("Posición (mm)")
            ax.set_ylabel("Radio (mm)")
            ax.set_title(f"Vista Superior - Geometría para Nota '{note}' - {self.flute_data.data.get('Flute Model', '')}")
            ax.grid(True)
            return fig
        except Exception as e:
            logger.error(f"Error generando vista superior de geometría para {self.flute_data.data.get('Flute Model')}, nota {note}: {e}")
            if 'fig' in locals(): plt.close(fig) # Cerrar la figura si se creó
            return None

    @staticmethod
    def plot_moc_summary(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]], 
            finger_frequencies_map: Dict[str, Dict[str, float]], # flute_name -> {note: freq}
            notes_ordered: List[str],
            base_colors: List[str] = BASE_COLORS, 
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7)) # Ajustar tamaño
        
        num_flutes = len(acoustic_analysis_list)
        offset_factor = 0.15
        if num_flutes > 1:
            offsets = np.linspace(-offset_factor * (num_flutes-1)/2, offset_factor * (num_flutes-1)/2, num_flutes)
        else:
            offsets = [0]
        
        legend_handles = []

        for index, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            style_idx = index % len(linestyles)
            color_idx = index % len(base_colors)
            linestyle = linestyles[style_idx]
            color = base_colors[color_idx]
            
            moc_values = []
            x_positions_for_plot = []
            current_finger_freqs = finger_frequencies_map.get(flute_name, {})

            for note_idx, note in enumerate(notes_ordered):
                moc = np.nan
                f_play = current_finger_freqs.get(note)

                if note in analysis_dict and f_play is not None:
                    analysis_obj = analysis_dict[note]
                    if isinstance(analysis_obj, ImpedanceComputation):
                        antiresonances = list(analysis_obj.antiresonance_frequencies())
                        if len(antiresonances) >= 2:
                            f0, f1 = antiresonances[0], antiresonances[1]
                            if f0 != 0 and f1 != 0 and f_play != 0 and f0 != f_play and f1 != (2*f_play): # Evitar divisiones por cero
                                # Fórmula original de MOC: ((1/f1) - (1/(2*fplay))) / ((1/f0) - (1/fplay))
                                # Numerador: (2*fplay - f1) / (2*f1*fplay)
                                # Denominador: (fplay - f0) / (f0*fplay)
                                # MOC = [(2*fplay - f1) / (2*f1*fplay)] * [(f0*fplay) / (fplay - f0)]
                                # MOC = [f0 * (2*fplay - f1)] / [2*f1 * (fplay - f0)]
                                numerator_term = (1 / f1) - (1 / (2 * f_play))
                                denominator_term = (1 / f0) - (1 / f_play)
                                if denominator_term != 0:
                                    moc = numerator_term / denominator_term
                moc_values.append(moc)
                x_positions_for_plot.append(note_idx + offsets[index])
            
            ax.plot(x_positions_for_plot, moc_values, marker="o", linestyle=linestyle, color=color)
            legend_handles.append(plt.Line2D([0], [0], color=color, linestyle=linestyle, marker='o', label=flute_name))

        ax.set_xticks(range(len(notes_ordered)))
        ax.set_xticklabels(notes_ordered)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
        ax.set_xlabel("Nota")
        ax.set_ylabel("MOC (ratio)")
        ax.set_title("Resumen de MOC por Nota")
        ax.grid(True)
        return fig
    
    @staticmethod
    def plot_bi_espe_summary(
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]], 
            finger_frequencies_map: Dict[str, Dict[str, float]], # flute_name -> {note: freq}
            notes_ordered: List[str],
            speed_of_sound: float = 343.0, # m/s, podría ser dependiente de la temperatura
            base_colors: List[str] = BASE_COLORS, 
            linestyles: List[str] = LINESTYLES
        ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(12, 7))
        num_flutes = len(acoustic_analysis_list)
        offset_factor = 0.1 # Más pequeño para BI y ESPE en el mismo x
        
        # Generar x_ticks una sola vez
        x_ticks_pos = np.arange(len(notes_ordered))

        for idx, (analysis_dict, flute_name) in enumerate(acoustic_analysis_list):
            color = base_colors[idx % len(base_colors)]
            # linestyle_bi = linestyles[idx % len(linestyles)] # Podríamos usar diferente para BI y ESPE
            # linestyle_espe = linestyles[(idx + len(linestyles)//2) % len(linestyles)]
            
            bi_values, espe_values = [], []
            current_finger_freqs = finger_frequencies_map.get(flute_name, {})

            for note in notes_ordered:
                bi, espe = np.nan, np.nan
                f_play_I = current_finger_freqs.get(note)

                if note in analysis_dict and f_play_I is not None:
                    analysis_obj = analysis_dict[note]
                    if isinstance(analysis_obj, ImpedanceComputation):
                        antiresonances = list(analysis_obj.antiresonance_frequencies())
                        if len(antiresonances) >= 2:
                            f0, f1 = antiresonances[0], antiresonances[1]
                            f_play_II = 2 * f_play_I
                            
                            if f0 > 0: bi = 1200 * np.log2(f_play_I / f0)
                            
                            delta_l_I = (speed_of_sound / 2) * ((1 / f_play_I) - (1 / f0)) if f_play_I > 0 and f0 > 0 else 0
                            delta_l_II = speed_of_sound * ((1 / f_play_II) - (1 / f1)) if f_play_II > 0 and f1 > 0 else 0
                            delta_delta_l = delta_l_II - delta_l_I
                            L_eff_I = (speed_of_sound / (2 * f_play_I)) if f_play_I > 0 else 0
                            
                            if L_eff_I > 0 and (L_eff_I + delta_delta_l) > 0: # Evitar log de <= 0
                                espe = 1200 * np.log2(L_eff_I / (L_eff_I + delta_delta_l))
                bi_values.append(bi)
                espe_values.append(espe)
            
            # Desplazamiento para esta flauta
            current_offset = (idx - num_flutes / 2 + 0.5) * offset_factor if num_flutes > 1 else 0
            
            ax.plot(x_ticks_pos + current_offset, bi_values, label=f"{flute_name} - $B_I$", linestyle='-', color=color, marker='o', markersize=5)
            ax.plot(x_ticks_pos + current_offset, espe_values, label=f"{flute_name} - ESPE", linestyle='--', color=color, marker='x', markersize=5)

        ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(notes_ordered)
        ax.set_title("$B_I$ y ESPE a Través de las Notas")
        ax.set_xlabel("Nota")
        ax.set_ylabel("Cents")
        ax.legend(fontsize=8, loc='best') # Ajustar loc y tamaño si hay muchas flautas
        ax.grid(True)
        fig.tight_layout()
        return fig

    # plot_summary_pdf no necesita ser estático si llama a métodos de instancia de FluteOperations.
    # Pero si los métodos que llama (plot_moc_summary, plot_bi_espe_summary) se vuelven estáticos,
    # entonces plot_summary_pdf también podría serlo o moverse a data_processing.py.
    # Por ahora, lo dejamos como método de instancia si se espera que 'self' sea una FluteOperations válida,
    # o lo hacemos estático y pasamos todo lo necesario.
    # Si es para generar un PDF de *una* flauta, es de instancia. Si es para *múltiples*, debería ser estático.
    # El original lo usaba con self.flute_ops_list[0].plot_moc_summary, así que lo hacemos estático.
    @staticmethod
    def generate_summary_pdf(
            pdf_filename: str,
            acoustic_analysis_list: List[Tuple[Dict[str, ImpedanceComputation], str]], 
            finger_frequencies_map: Dict[str, Dict[str, float]], 
            notes_ordered: List[str],
            # speed_of_sound: float = 343.0 # Si es necesario para B_I/ESPE
        ) -> str:
        """
        Genera un reporte PDF incluyendo el resumen de MOC y el resumen de B_I/ESPE.
        """
        with PdfPages(pdf_filename) as pdf:
            logger.info(f"Generando gráfico MOC para PDF: {pdf_filename}")
            fig_moc = FluteOperations.plot_moc_summary(acoustic_analysis_list, finger_frequencies_map, notes_ordered)
            pdf.savefig(fig_moc)
            plt.close(fig_moc)
            logger.info("Gráfico MOC guardado en PDF.")

            logger.info(f"Generando gráfico B_I/ESPE para PDF: {pdf_filename}")
            # Asumir speed_of_sound por defecto o pasarlo como argumento si varía (ej. por temperatura)
            fig_bi_espe = FluteOperations.plot_bi_espe_summary(acoustic_analysis_list, finger_frequencies_map, notes_ordered)
            pdf.savefig(fig_bi_espe)
            plt.close(fig_bi_espe)
            logger.info("Gráfico B_I/ESPE guardado en PDF.")
        
        logger.info(f"Reporte PDF de resumen guardado en: {pdf_filename}")
        return pdf_filename