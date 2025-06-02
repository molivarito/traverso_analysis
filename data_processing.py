import json # No usado directamente aquí
import tempfile # No usado directamente aquí
from pathlib import Path
from typing import List, Tuple, Any, Dict # Any -> Dict

import numpy as np # No usado directamente aquí
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Asumiendo que FluteData y FluteOperations están disponibles en el path
# from flute_data import FluteData # No se instancia aquí
from flute_operations import FluteOperations # Para llamar métodos estáticos
from constants import BASE_COLORS, COLORMAP_LARGE_NUMBER_OF_FLUTES, LINESTYLES # Importar constantes

import logging

logger = logging.getLogger(__name__)

# DEFAULT_FING_CHART_PATH ya no se usa aquí directamente, se obtiene de FluteData si es necesario
# SCRIPT_DIR = Path(__file__).resolve().parent # No necesario si no se construye DEFAULT_FING_CHART_PATH aquí
# DEFAULT_FING_CHART_PATH = SCRIPT_DIR.parent / "data_json" / "traverso_fingerchart.txt"
# if not DEFAULT_FING_CHART_PATH.exists():
#     DEFAULT_FING_CHART_PATH = Path("data_json") / "traverso_fingerchart.txt"


def save_plots_to_pdf(
        flute_operations_list: List[FluteOperations], # Lista de instancias de FluteOperations
        output_pdf_paths: Tuple[str, str],
        # fing_chart_file_path: str = str(DEFAULT_FING_CHART_PATH) # Obtenido de FluteData
    ) -> None:
    geometrical_pdf_path, acoustic_pdf_path = output_pdf_paths

    try:
        flute_names = [fo.flute_data.data.get("Flute Model", f"UnknownFlute_{i}") for i, fo in enumerate(flute_operations_list)]
        num_flutes = len(flute_operations_list)

        # Generar colores y estilos (similar a como estaba, pero usando constantes)
        if num_flutes > len(BASE_COLORS):
            colormap = plt.cm.get_cmap(COLORMAP_LARGE_NUMBER_OF_FLUTES, num_flutes)
            plot_colors = [colormap(i) for i in range(num_flutes)]
        else:
            plot_colors = [BASE_COLORS[i % len(BASE_COLORS)] for i in range(num_flutes)]
        
        plot_styles = [LINESTYLES[i % len(LINESTYLES)] for i in range(num_flutes)]

        # --- Guardar gráficos geométricos (mayormente de instancia) ---
        geom_pdf_path_obj = Path(geometrical_pdf_path)
        with PdfPages(geom_pdf_path_obj) as geometrical_pdf:
            logger.info("Generando gráficos geométricos...")

            # Gráficos individuales de partes (requiere un subplot por flauta o manejo complejo)
            # Esta parte es compleja si se quieren todas las flautas en una sola figura de subplots.
            # Por simplicidad, graficaremos las partes de cada flauta en su propia figura.
            logger.info("Plotting individual parts for each flute...")
            for idx, fo in enumerate(flute_operations_list):
                fig_individual, _ = fo.plot_individual_parts(
                    flute_names=[flute_names[idx]], 
                    flute_color=plot_colors[idx]
                )
                geometrical_pdf.savefig(fig_individual)
                plt.close(fig_individual)

            # Partes superpuestas (plot_all_parts_overlapping)
            logger.info("Plotting overlapping parts (sequential display)...")
            fig_overlapping_seq, ax_overlapping_seq = plt.subplots(figsize=(12,6)) # Ajustar según necesidad
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_all_parts_overlapping(
                    ax=ax_overlapping_seq, 
                    flute_names=[flute_names[idx]], 
                    flute_color=plot_colors[idx], 
                    flute_style=plot_styles[idx]
                )
            ax_overlapping_seq.legend(loc='best', title="Flautas") # Añadir leyenda general
            geometrical_pdf.savefig(fig_overlapping_seq)
            plt.close(fig_overlapping_seq)

            # Perfil combinado (plot_combined_flute_data)
            logger.info("Plotting combined flute profiles...")
            fig_combined_profile, ax_combined_profile = plt.subplots(figsize=(12,6))
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_combined_flute_data(
                    ax=ax_combined_profile, 
                    flute_names=[flute_names[idx]], 
                    flute_color=plot_colors[idx], 
                    flute_style=plot_styles[idx]
                )
            ax_combined_profile.legend(loc='best', title="Flautas")
            geometrical_pdf.savefig(fig_combined_profile)
            plt.close(fig_combined_profile)

            # Vista 2D (plot_flute_2d_view)
            logger.info("Plotting 2D views...")
            fig_2d_view, ax_2d_view = plt.subplots(figsize=(15,5)) # Ajustar
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_flute_2d_view(
                    ax=ax_2d_view, 
                    flute_names=[flute_names[idx]], 
                    flute_color=plot_colors[idx], 
                    flute_style=plot_styles[idx]
                )
            ax_2d_view.legend(loc='best', title="Flautas")
            geometrical_pdf.savefig(fig_2d_view)
            plt.close(fig_2d_view)
            
            # Geometría del instrumento (plot_instrument_geometry) y Top View (plot_top_view_instrument_geometry)
            # Estos son por nota. Podríamos elegir una nota representativa o iterar.
            # Aquí se muestra para una nota "D" por defecto.
            default_note_for_geom = "D"
            logger.info(f"Plotting instrument geometry for note {default_note_for_geom}...")
            for idx, fo in enumerate(flute_operations_list):
                fig_geom_inst = fo.plot_instrument_geometry(note=default_note_for_geom) # Devuelve fig, ax o solo ax
                if fig_geom_inst: # Si devuelve figura
                     geometrical_pdf.savefig(fig_geom_inst)
                     plt.close(fig_geom_inst)
                elif isinstance(fig_geom_inst, plt.Axes): # Si devuelve solo ejes
                     geometrical_pdf.savefig(fig_geom_inst.figure)
                     plt.close(fig_geom_inst.figure)


                fig_top_view = fo.plot_top_view_instrument_geometry(note=default_note_for_geom)
                if fig_top_view:
                    geometrical_pdf.savefig(fig_top_view)
                    plt.close(fig_top_view)
            
            logger.info("Gráficos geométricos guardados en: %s", geom_pdf_path_obj)

        # --- Guardar gráficos acústicos (usando métodos estáticos donde aplique) ---
        acoustic_pdf_path_obj = Path(acoustic_pdf_path)
        with PdfPages(acoustic_pdf_path_obj) as acoustic_pdf:
            logger.info("Generando gráficos acústicos...")

            # Lista de análisis acústicos y nombres para métodos estáticos
            acoustic_analysis_data_list = [
                (fo.flute_data.acoustic_analysis, fo.flute_data.data.get("Flute Model", f"Unknown_{i}")) 
                for i, fo in enumerate(flute_operations_list)
            ]
            # Mapa de frecuencias de digitación por flauta
            finger_frequencies_map_data = {
                fo.flute_data.data.get("Flute Model", f"Unknown_{i}"): fo.flute_data.finger_frequencies
                for i, fo in enumerate(flute_operations_list)
            }

            # Necesitamos una lista ordenada común de notas.
            # Tomar del primer archivo de digitación o construir un conjunto de todas las flautas.
            # Asumimos que el fing_chart_file_path es el mismo o compatible para todas las flautas cargadas
            # o que las notas relevantes están en FluteData.finger_frequencies.keys()
            
            # Obtener notas comunes de finger_frequencies
            common_notes_set = set()
            if flute_operations_list:
                 # Tomar el orden de la primera flauta como referencia, luego filtrar las que son comunes a todas
                # o usar todas las notas presentes en finger_frequencies_map_data
                # Para simplificar, usamos las notas de la primera flauta como base del orden.
                # Una lógica más robusta buscaría la intersección o unión y ordenaría canónicamente.
                # fing_chart_path_ref = flute_operations_list[0].flute_data.fing_chart_file # No existe este atributo
                # Necesitamos el fing_chart_file_path pasado como argumento.
                # Por ahora, si no se pasa, no podemos obtener el orden canónico.
                # Usaremos las claves de finger_frequencies de la primera flauta.
                
                first_flute_notes = list(flute_operations_list[0].flute_data.finger_frequencies.keys())
                # Intentar ordenar canónicamente (ej. D, E, Fs, G, A, B, Cs) si es posible
                # Si no, el orden de `keys()` puede variar.
                # Definir un orden canónico si es conocido:
                canonical_note_order = ["D", "D#", "E", "F", "Fs", "G", "G#", "A", "A#", "B", "C", "Cs"] # Ajustar a las notas reales
                
                # Filtrar y ordenar notas presentes
                all_present_notes = set()
                for ff_map in finger_frequencies_map_data.values():
                    all_present_notes.update(ff_map.keys())
                
                ordered_notes_for_plots = [n for n in canonical_note_order if n in all_present_notes]
                # Añadir notas no canónicas al final
                ordered_notes_for_plots.extend(sorted(list(all_present_notes - set(ordered_notes_for_plots))))

            else:
                ordered_notes_for_plots = []


            if not ordered_notes_for_plots:
                logger.warning("No se pudieron determinar notas ordenadas para los gráficos de resumen acústico.")
            
            # Admitancia combinada
            logger.info("Plotting combined admittance...")
            fig_combined_adm = FluteOperations.plot_combined_admittance(acoustic_analysis_data_list)
            acoustic_pdf.savefig(fig_combined_adm)
            plt.close(fig_combined_adm)

            # Frecuencias antiresonantes
            if ordered_notes_for_plots:
                logger.info("Plotting summary of antiresonances...")
                fig_antiresonances = FluteOperations.plot_summary_antiresonances(acoustic_analysis_data_list, ordered_notes_for_plots)
                acoustic_pdf.savefig(fig_antiresonances)
                plt.close(fig_antiresonances)
            
                # Diferencias en cents
                logger.info("Plotting summary of cents differences...")
                fig_cents = FluteOperations.plot_summary_cents_differences(acoustic_analysis_data_list, ordered_notes_for_plots)
                acoustic_pdf.savefig(fig_cents)
                plt.close(fig_cents)
            
                # Gráfico de MOC
                logger.info("Plotting MOC summary...")
                fig_moc = FluteOperations.plot_moc_summary(acoustic_analysis_data_list, finger_frequencies_map_data, ordered_notes_for_plots)
                acoustic_pdf.savefig(fig_moc)
                plt.close(fig_moc)

                # Gráfico de BI y ESPE
                logger.info("Plotting B_I and ESPE summary...")
                fig_bi_espe = FluteOperations.plot_bi_espe_summary(acoustic_analysis_data_list, finger_frequencies_map_data, ordered_notes_for_plots)
                acoustic_pdf.savefig(fig_bi_espe)
                plt.close(fig_bi_espe)

            # Admitancia individual por nota
            # Iterar sobre notas comunes o todas las notas presentes.
            # 'ordered_notes_for_plots' es una buena lista para esto.
            if ordered_notes_for_plots:
                logger.info("Plotting individual admittance analysis per note...")
                for note_to_plot in ordered_notes_for_plots:
                    try:
                        fig_indiv_adm = FluteOperations.plot_individual_admittance_analysis(acoustic_analysis_data_list, note_to_plot)
                        acoustic_pdf.savefig(fig_indiv_adm)
                        plt.close(fig_indiv_adm)
                    except Exception as e_indiv_adm:
                        logger.error(f"Error graficando admitancia individual para nota {note_to_plot}: {e_indiv_adm}")


            logger.info("Gráficos acústicos guardados en: %s", acoustic_pdf_path_obj)
    except Exception as e:
        logger.exception("Error al guardar gráficos en los PDF: %s", e)