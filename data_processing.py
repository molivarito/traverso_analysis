import json
import tempfile
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from flute_data import FluteData
from flute_operations import FluteOperations
from notion_utils import get_json_files_from_notion

import logging

logger = logging.getLogger(__name__)

DEFAULT_FING_CHART_PATH = "data_json/traverso_fingerchart.txt"

def save_plots_to_pdf(flute_operations_list: List[FluteOperations], output_pdf_paths: Tuple[str, str], fing_chart_file_path: str = DEFAULT_FING_CHART_PATH) -> None:
    """
    Genera y guarda gráficos combinados de múltiples flautas en archivos PDF.

    Args:
        flute_operations_list (List[FluteOperations]): Lista de objetos FluteOperations.
        fing_chart_file_path (str): Ruta al archivo de digitaciones.
        output_pdf_paths (tuple): Rutas de los archivos PDF (geométricos, acústicos).
    """
    geometrical_pdf_path, acoustic_pdf_path = output_pdf_paths

    try:
        # Obtener nombres de las flautas
        flute_names = [fo.flute_data.data.get("Flute Model", "Unknown") for fo in flute_operations_list]

        # Generar colores y estilos
        num_flutes = len(flute_operations_list)
        # Usar un colormap para más variedad si hay muchas flautas
        if num_flutes > 6: # Ejemplo, podrías usar tab10 o tab20
            colormap = plt.cm.get_cmap('tab10', num_flutes) # 'viridis', 'tab10', 'tab20'
            flute_colors = [colormap(i) for i in range(num_flutes)]
        else:
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # Colores por defecto de Matplotlib
            flute_colors = [base_colors[i % len(base_colors)] for i in range(num_flutes)]
        flute_styles = ['-', '--', '-.', ':'] * (num_flutes // 4 + 1) # Repetir estilos

        # --- Guardar gráficos geométricos ---
        geom_pdf_path = Path(geometrical_pdf_path)
        with PdfPages(geom_pdf_path) as geometrical_pdf:
            logger.info("Generando gráficos geométricos...")

            # Gráficos individuales
            fig_individual, ax_individual = plt.subplots(2, 2, figsize=(14, 8))
            ax_individual = ax_individual.flatten()
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_individual_parts(ax=ax_individual, flute_names=[flute_names[idx]], flute_color=flute_colors[idx])
            geometrical_pdf.savefig(fig_individual)
            plt.close(fig_individual)

            # Partes superpuestas
            fig_overlapping, ax_overlapping = plt.subplots()
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_all_parts_overlapping(ax=ax_overlapping, flute_names=[flute_names[idx]], flute_color=flute_colors[idx], flute_style=flute_styles[idx])
            geometrical_pdf.savefig(fig_overlapping)
            plt.close(fig_overlapping)

            # Perfil combinado
            fig_combined, ax_combined = plt.subplots()
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_combined_flute_data(ax=ax_combined, flute_names=[flute_names[idx]], flute_color=flute_colors[idx], flute_style=flute_styles[idx])
            geometrical_pdf.savefig(fig_combined)
            plt.close(fig_combined)

            # Vista 2D
            fig_2d, ax_2d = plt.subplots()
            for idx, fo in enumerate(flute_operations_list):
                fo.plot_flute_2d_view(ax=ax_2d, flute_names=[flute_names[idx]], flute_color=flute_colors[idx], flute_style=flute_styles[idx])
            geometrical_pdf.savefig(fig_2d)
            plt.close(fig_2d)

            # Geometría del instrumento
            for fo in flute_operations_list:
                fig_geometry = fo.plot_instrument_geometry()
                if fig_geometry:
                    geometrical_pdf.savefig(fig_geometry)
                    plt.close(fig_geometry)

            logger.info("Gráficos geométricos guardados en: %s", geometrical_pdf_path)

        # --- Guardar gráficos acústicos ---
        acoustic_pdf_path_obj = Path(acoustic_pdf_path)
        with PdfPages(acoustic_pdf_path_obj) as acoustic_pdf:
            logger.info("Generando gráficos acústicos...")

            acoustic_analysis_list = [
                (fo.flute_data.acoustic_analysis, fo.flute_data.data.get("Flute Model", "Unknown"))
                for fo in flute_operations_list
            ]

            # Geometría con fingering
            for fo in flute_operations_list:
                try:
                    fig_top_view = fo.plot_top_view_instrument_geometry(note="D")
                    if fig_top_view:
                        acoustic_pdf.savefig(fig_top_view)
                        plt.close(fig_top_view)
                        logger.info("Geometría guardada para: %s", fo.flute_data.data.get("Flute Model", "Unknown"))
                except Exception as e:
                    logger.exception("Error al graficar la geometría: %s", e)

            # Orden de notas
            fing_chart_file = Path(fing_chart_file_path)
            with fing_chart_file.open("r") as file:
                header_line = file.readline().strip() # Asumiendo que las notas están en la primera línea
                ordered_notes = header_line.split()[1:] # Omitir "label"

            common_notes = set()
            for analysis, _ in acoustic_analysis_list:
                common_notes.update(analysis.keys())
            common_notes = [note for note in ordered_notes if note in common_notes]
            
            # Admitancia combinada
            fig_combined_adm = flute_operations_list[0].plot_combined_admittance(acoustic_analysis_list) # Usar método público
            acoustic_pdf.savefig(fig_combined_adm)
            plt.close(fig_combined_adm)

            # Frecuencias antiresonantes
            fig_antiresonances = flute_operations_list[0].plot_summary_antiresonances(acoustic_analysis_list, common_notes) # Usar método público
            acoustic_pdf.savefig(fig_antiresonances)
            plt.close(fig_antiresonances)
            # Diferencias en cents
            fig_cents = flute_operations_list[0].plot_summary_cents_differences(acoustic_analysis_list, common_notes) # Usar método público
            acoustic_pdf.savefig(fig_cents)
            plt.close(fig_cents)
            # Gráfico de MOC
            fig_moc = flute_operations_list[0].plot_moc_summary(acoustic_analysis_list, flute_operations_list[0].flute_data.finger_frequencies)
            acoustic_pdf.savefig(fig_moc)
            plt.close(fig_moc)

            # Gráfico de BI y ESPE
            fig_bi_espe = flute_operations_list[0].plot_bi_espe_summary(acoustic_analysis_list, flute_operations_list[0].flute_data.finger_frequencies)
            acoustic_pdf.savefig(fig_bi_espe)
            plt.close(fig_bi_espe)

            # Admitancia individual por nota
            for note in common_notes:
                fig_indiv = flute_operations_list[0].plot_individual_admittance_analysis(acoustic_analysis_list, note) # Usar método público
                acoustic_pdf.savefig(fig_indiv)
                plt.close(fig_indiv)


            logger.info("Gráficos acústicos guardados en: %s", acoustic_pdf_path)
    except Exception as e:
        logger.exception("Error al guardar gráficos en los PDF: %s", e)