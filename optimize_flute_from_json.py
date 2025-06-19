#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_flute_from_json.py

Optimiza la altura de la chimenea de la embocadura para que una flauta
suene afinada a un diapasón específico.
Precalcula datos de presión/flujo.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import copy
from typing import Optional, List, Dict, Any, Tuple
from openwind import ImpedanceComputation # <--- AÑADIR ESTA LÍNEA
from openwind import InstrumentGeometry, InstrumentPhysics, Player
from openwind.inversion import InverseFrequentialResponse

try:
    from flute_data import FluteData, FLUTE_PARTS_ORDER # Importar FLUTE_PARTS_ORDER
    from constants import MM_TO_M_FACTOR as MM_TO_M_FACTOR_CONST, M_TO_MM_FACTOR as M_TO_MM_FACTOR_CONST
except ImportError as e:
    print(f"Error importando módulos locales: {e}. Asegúrate de que flute_data.py y constants.py están accesibles.")
    exit()

MM_TO_M_FACTOR = 0.001 # Constante local para evitar dependencia si constants.py no está
M_TO_MM_FACTOR = 1000.0


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s [%(name)s:%(funcName)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

DIAPASON_A4_HZ = 415.0
TARGET_TEMPERATURE_C = 25.0
OPTIM_CHIMNEY_RANGE_MM_STR = "1.0<~10.0"
INITIAL_CHIMNEY_HEIGHT_M_FOR_OPTIM = 3.0 * MM_TO_M_FACTOR
EMBOUCHURE_HOLE_LABEL = "embouchure"


def plot_optimized_admittances(admittance_data: dict, flute_model_name: str, target_freqs: dict, diapason: float, return_fig: bool = False):
    num_notes = len(admittance_data)
    if num_notes == 0:
        logger.info("No hay datos de admitancia para plotear.")
        fig_empty, ax_empty = plt.subplots()
        ax_empty.text(0.5, 0.5, "No admittance data", ha='center', va='center')
        return fig_empty if return_fig else None

    cols = 2
    rows = (num_notes + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(7*cols, 5*rows), squeeze=False)
    axs_flat = axs.flatten()

    note_idx = 0
    for note_name, (freqs, adm_db) in admittance_data.items():
        if note_idx >= len(axs_flat): break

        ax = axs_flat[note_idx]
        if freqs.size > 0 and adm_db.size > 0:
            ax.plot(freqs, adm_db, label="Admitancia Optimizada")
            if note_name in target_freqs:
                ax.axvline(target_freqs[note_name], color='r', linestyle='--', label=f"Frec. Obj: {target_freqs[note_name]:.1f} Hz")

            from scipy.signal import find_peaks # Importación local por si no es global
            peaks, _ = find_peaks(adm_db, height=np.max(adm_db)-20 if adm_db.size >0 else None)
            if peaks.size > 0:
                peak_freqs = freqs[peaks]
                peak_adms = adm_db[peaks]
                ax.plot(peak_freqs, peak_adms, "x", color='g', label="Picos Admitancia")
                for i, pf in enumerate(peak_freqs[:2]):
                    ax.text(pf, peak_adms[i], f"{pf:.0f}", color='g', ha='center', va='bottom')

        ax.set_title(f"Nota: {note_name}")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Admitancia (dB)")
        ax.legend(fontsize='small')
        ax.grid(True, linestyle=':')
        note_idx += 1

    for i in range(note_idx, len(axs_flat)):
        axs_flat[i].set_visible(False)

    fig.suptitle(f"Admitancias Optimizadas para {flute_model_name} (A4={diapason}Hz)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if return_fig:
        return fig
    # else: # Si no se devuelve la figura, se podría mostrar o cerrar aquí.
    #     plt.show() # O plt.close(fig) si no se quiere mostrar interactivamente.
    return None


def optimize_flute_from_json_full(
    diapason_a4_hz_gui: float,
    target_temp_c_gui: float,
    flute_data_instance: Optional[FluteData] = None, # type: ignore
    flute_dir_path_str: Optional[str] = None
) -> tuple[dict, dict, dict, str, dict, float, dict, dict, list, dict]:
    """ # Añadido 'dict' al final para el análisis inicial
    Carga BORE, HOLES y CHART desde JSON (si no se pasa flute_data_instance) o usa la instancia proporcionada.
    Devuelve: optimized_chimney_heights_mm, initial_admittance_data, optimized_admittance_data, flute_model_name,
              target_frequencies_hz_map, diapason_a4_hz_gui, physics_states_per_note (solo geometría),
              pressure_flow_data_per_note (precalculado), notes_optimized_list
    """ # noqa: E501, E261

    current_flute_data_instance: Optional[FluteData] = flute_data_instance

    # --- Cargar FluteData si no se proporciona una instancia ---
    if current_flute_data_instance is None:
        if not flute_dir_path_str:
            logger.error("Se debe proporcionar 'flute_data_instance' o 'flute_dir_path_str'. Ninguno fue dado.")
            return {}, {}, {}, "", {}, 0.0, {}, {}, [], {}

        flute_dir_path = Path(flute_dir_path_str)
        if not flute_dir_path.is_dir(): # type: ignore
            logger.error(f"El directorio proporcionado no existe o no es un directorio: {flute_dir_path}")
            return {}, {}, {}, "", {}, 0.0, {}, {}, []
        try:
            logger.info(f"Cargando FluteData desde la ruta: {flute_dir_path}")
            current_flute_data_instance = FluteData(
                str(flute_dir_path),
                la_frequency=diapason_a4_hz_gui,
                temperature=target_temp_c_gui,
                skip_acoustic_analysis=False # El análisis es necesario para la optimización
            )
            if current_flute_data_instance.validation_errors:
                logger.error(f"Errores de validación al cargar FluteData desde '{flute_dir_path}': {current_flute_data_instance.validation_errors}")
                # Retornar con errores si la validación falla, incluyendo el análisis inicial si se calculó algo
                return {}, {}, {}, current_flute_data_instance.flute_model, {}, diapason_a4_hz_gui, {}, {}, [], {}
        except Exception as e:
            logger.error(f"Error al cargar FluteData desde '{flute_dir_path}': {e}", exc_info=True)
            return {}, {}, {}, "", {}, 0.0, {}, {}, []
    # --- Fin Carga FluteData ---
    if not current_flute_data_instance: # Si después de todo, sigue siendo None
        logger.error("Falló la obtención o carga de flute_data_instance.")
        return {}, {}, {}, "", {}, 0.0, {}, {}, [], {}

    flute_model_name = current_flute_data_instance.flute_model
    logger.info(f"Usando datos de FluteData para '{flute_model_name}' para optimización.")

    # Guardar el análisis inicial completo antes de modificar la geometría para la optimización
    initial_acoustic_analysis_dict = copy.deepcopy(current_flute_data_instance.acoustic_analysis)

    # Obtener inputs de geometría para OpenWind (estos ya reflejan la geometría inicial)
    bore_m_radius, side_holes_m_data_raw, fing_chart_ow_from_json = current_flute_data_instance.get_openwind_geometry_inputs()

    if not bore_m_radius:
        logger.error(f"No se pudo obtener el bore de FluteData para {flute_model_name}")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe
    if not side_holes_m_data_raw:
        logger.error(f"No se pudieron obtener los datos de agujeros de FluteData para {flute_model_name}")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe
    if not fing_chart_ow_from_json:
        logger.error(f"No se pudo obtener la tabla de digitaciones de FluteData para {flute_model_name}")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe

    # Preparar datos de agujeros para InstrumentGeometry, marcando la embocadura para optimización
    bore_mm_diam_from_json = []
    for seg in bore_m_radius:
        bore_mm_diam_from_json.append([
            seg[0] * M_TO_MM_FACTOR, seg[1] * M_TO_MM_FACTOR,
            seg[2] * 2 * M_TO_MM_FACTOR, seg[3] * 2 * M_TO_MM_FACTOR,
            seg[4]
        ])
    logger.info(f"Bore procesado desde JSON para '{flute_model_name}'.")

    side_holes_header_mm_diam = [['label', 'location', 'chimney', 'diameter']]
    side_holes_mm_diam_data_from_json = side_holes_header_mm_diam

    found_embouchure_for_optim = False
    for hole_m_entry in side_holes_m_data_raw[1:]:
        label, pos_m, chimney_m, radius_m, radius_out_m = hole_m_entry
        chimney_val_mm = chimney_m * M_TO_MM_FACTOR
        diameter_val_mm = radius_m * 2 * M_TO_MM_FACTOR

        if str(label).lower() == EMBOUCHURE_HOLE_LABEL:
            logger.info(f"Embocadura original (m) desde JSON: pos={pos_m:.4f}, chimenea={chimney_m:.4f}, radio={radius_m:.4f}")
            chimney_input_for_ig = OPTIM_CHIMNEY_RANGE_MM_STR
            found_embouchure_for_optim = True
            logger.info(f"  Embocadura '{label}' (JSON) marcada para optimización con rango de chimenea: '{chimney_input_for_ig}' (mm)")
        else:
            chimney_input_for_ig = chimney_val_mm
        side_holes_mm_diam_data_from_json.append([
            label,
            pos_m * M_TO_MM_FACTOR,
            chimney_input_for_ig,
            diameter_val_mm
        ])

    if not found_embouchure_for_optim:
        logger.error(f"No se encontró la entrada '{EMBOUCHURE_HOLE_LABEL}' en los datos de agujeros de JSON para {flute_model_name}. No se puede optimizar.")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, []
    logger.info(f"Agujeros procesados para InstrumentGeometry para '{flute_model_name}'.")

    fing_chart_ow_filtered = fing_chart_ow_from_json
    if fing_chart_ow_from_json and len(fing_chart_ow_from_json) > 1:
        temp_filtered_chart = [fing_chart_ow_from_json[0]]
        for row in fing_chart_ow_from_json[1:]:
            if row[0].lower() != EMBOUCHURE_HOLE_LABEL:
                temp_filtered_chart.append(row)
        fing_chart_ow_filtered = temp_filtered_chart
    logger.info(f"Tabla de digitaciones procesada para InstrumentGeometry para '{flute_model_name}'.")

    try:
        my_geom = InstrumentGeometry(bore_mm_diam_from_json, side_holes_mm_diam_data_from_json, fing_chart_ow_filtered,
                                     unit='mm', diameter=True)
        logger.debug(f"InstrumentGeometry creada para '{flute_model_name}'")

        if not my_geom.optim_params or not my_geom.optim_params.labels:
            logger.error(f"No se encontraron parámetros optimizables en InstrumentGeometry para '{flute_model_name}'.")
            return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, []

        actual_optim_param_label = my_geom.optim_params.labels[0]
        logger.info(f"Parámetro optimizable confirmado: '{actual_optim_param_label}' (Esperado: {EMBOUCHURE_HOLE_LABEL}_chimney)")
        if actual_optim_param_label != f"{EMBOUCHURE_HOLE_LABEL}_chimney":
            logger.warning(f"El parámetro optimizable detectado '{actual_optim_param_label}' no es el esperado '{EMBOUCHURE_HOLE_LABEL}_chimney'.")

    except Exception as e:
        logger.error(f"Error al crear InstrumentGeometry para '{flute_model_name}': {e}", exc_info=True)
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe

    notes_to_optimize = my_geom.fingering_chart.all_notes()
    if not notes_to_optimize:
        logger.error(f"No hay notas en la tabla de digitaciones procesada para {flute_model_name}.")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe

    target_frequencies_hz_map = {}
    if current_flute_data_instance.finger_frequencies:
        for note in notes_to_optimize:
            if note in current_flute_data_instance.finger_frequencies:
                target_frequencies_hz_map[note] = current_flute_data_instance.finger_frequencies[note]
            else:
                logger.warning(f"Nota '{note}' del chart no encontrada en finger_frequencies de FluteData para {flute_model_name}. Saltando esta nota.")
    else:
        logger.error(f"finger_frequencies no está disponible en FluteData para {flute_model_name}. No se pueden determinar frecuencias objetivo.")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, []
 # Retornar análisis inicial si existe
    if not target_frequencies_hz_map:
        logger.error(f"No hay notas con frecuencias objetivo válidas para optimizar en {flute_model_name}.")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, []

    logger.info(f"Notas a optimizar y sus frecuencias objetivo (Hz): {target_frequencies_hz_map}")

    player = Player()
    physics_options = {
        'temperature': target_temp_c_gui, 'player': player,
        'source_location': EMBOUCHURE_HOLE_LABEL,
        'losses': True, 'humidity': 0.5,
        'radiation_category': {'entrance':'closed', 'holes':'unflanged', 'bell':'unflanged'}
    }
    try:
        my_phy = InstrumentPhysics(instrument_geometry=my_geom, **physics_options)
    except Exception as e:
        logger.error(f"Error al inicializar InstrumentPhysics para '{flute_model_name}': {e}", exc_info=True)
        return {}, {}, {}, flute_model_name, target_frequencies_hz_map, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe

    Z_target_for_optim = np.array([0])
    if not target_frequencies_hz_map or not any(target_frequencies_hz_map.values()):
        logger.error(f"No hay frecuencias objetivo para inicializar optim_tool en {flute_model_name}.")
        return {}, {}, {}, flute_model_name, {}, diapason_a4_hz_gui, {}, {}, [], initial_acoustic_analysis_dict # Retornar análisis inicial si existe
    initial_freq_for_optim_tool = list(target_frequencies_hz_map.values())[0]

    # Crear el optimizador
    optim_tool = InverseFrequentialResponse(my_phy,
                                            initial_freq_for_optim_tool,
                                            [Z_target_for_optim],
                                            observable='reflection')

    optimized_chimney_heights_mm: Dict[str, float] = {}
    initial_admittance_data = {}
    optimized_admittance_data = {}
    pressure_flow_data_per_note: Dict[str, Dict[str, Any]] = {}
    optimized_impedance_computations_per_note: Dict[str, ImpedanceComputation] = {} # Nuevo diccionario para ImpedanceComputation optimizados
    notes_actually_optimized = []

    for note_name in notes_to_optimize:
        if note_name not in target_frequencies_hz_map:
            logger.warning(f"Nota {note_name} no tiene frecuencia objetivo definida. Saltando.")
            continue

        target_freq = target_frequencies_hz_map[note_name]
        logger.info(f"\n--- Optimizando para Nota: {note_name}, Frec. Objetivo: {target_freq:.2f} Hz ---")
        logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Inicio del bucle de optimización.")

        try:
            my_geom.optim_params.set_active_values([INITIAL_CHIMNEY_HEIGHT_M_FOR_OPTIM])
            current_val_m = my_geom.optim_params.get_active_values()[0]
            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Altura de chimenea reseteada a (m): {current_val_m}")
            logger.info(f"Altura inicial de chimenea para '{note_name}' (parámetro '{actual_optim_param_label}'): {current_val_m * M_TO_MM_FACTOR:.3f} mm")
        except Exception as e_set_param:
            notes_actually_optimized.append(note_name) # Añadir nota incluso si falla el reset para que aparezca en la lista
            logger.error(f"Error reseteando parámetro optimizable para nota {note_name}: {e_set_param}", exc_info=True)
            optimized_chimney_heights_mm[note_name] = np.nan
            continue

        try:
            optim_tool.set_note(note_name)
            freq_simu_admittance = np.arange(100, 5002, 2.0)
            optim_tool.update_frequencies_and_mesh(freq_simu_admittance)
            optim_tool.solve()
            impedance_values_initial = optim_tool.impedance
            valid_impedance_initial = np.where(np.abs(impedance_values_initial) < 1e-12, 1e-12, impedance_values_initial)
            admittance_db_initial = 20 * np.log10(np.abs(1.0 / valid_impedance_initial))
            initial_admittance_data[note_name] = (optim_tool.frequencies.copy(), admittance_db_initial.copy())
            logger.info(f"  Admitancia inicial calculada para {note_name}")

            optim_tool.update_frequencies_and_mesh(np.array([target_freq]))
            optim_tool.set_targets_list([Z_target_for_optim], [note_name])

            logger.info(f"Iniciando optimización para {note_name}...")
            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Antes de optim_tool.optimize_freq_model.")
            optim_tool.optimize_freq_model(iter_detailed=True)
            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Después de optim_tool.optimize_freq_model.")

            optimized_value_m = my_geom.optim_params.get_active_values()[0] # Get the optimized value from the original geom object
            optimized_chimney_heights_mm[note_name] = optimized_value_m * M_TO_MM_FACTOR
            logger.info(f"Nota {note_name}: Altura de chimenea optimizada = {optimized_chimney_heights_mm[note_name]:.3f} mm")
            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Altura chimenea optimizada (m): {optimized_value_m}")

            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Antes de recompute_impedance_at.")
            optim_tool.recompute_impedance_at(freq_simu_admittance)
            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Después de recompute_impedance_at.")

            impedance_values = optim_tool.impedance
            valid_impedance = np.where(np.abs(impedance_values) < 1e-12, 1e-12, impedance_values)
            admittance_db = 20 * np.log10(np.abs(1.0 / valid_impedance))
            optimized_admittance_data[note_name] = (optim_tool.frequencies.copy(), admittance_db.copy())

            # Inspeccionar el objeto impedance_computation
            f_res_optim = optim_tool.antiresonance_frequencies(k=2, display_warning=False)
            if len(f_res_optim) >= 1:
                # Check if f_res_optim has at least one element before accessing f_res_optim[0]
                first_ar_freq = f_res_optim[0] if f_res_optim.size > 0 else np.nan
                logger.info(f'  Final deviation: {first_ar_freq} vs {target_freq}')
                logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Frec. antirresonancia optimizada (desde optim_tool.antiresonance_frequencies): {first_ar_freq if not np.isnan(first_ar_freq) else 'N/A'}")
            if len(f_res_optim) >= 2 and f_res_optim[0] != 0:
                logger.info(f'  Final harmonicity: {f_res_optim[1]/f_res_optim[0]}')

            # Create a NEW InstrumentGeometry object with the optimized chimney height for this note.
            # This avoids relying on the potentially unstable optim_tool.instru_physics.geom state.
            final_optimized_ic_for_storage = None
            try:
                logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Intentando crear ImpedanceComputation final para almacenamiento.")
                # Get the original geometry inputs
                _original_bore_m_radius_segments, side_holes_m_data_raw_original, fing_chart_ow_from_json_original = current_flute_data_instance.get_openwind_geometry_inputs()

                # Preparar main_bore para ImpedanceComputation (lista de puntos [x,r] en metros, relativos al corcho)
                # El bore no cambia con la optimización de la chimenea, usamos el original de FluteData.
                stopper_offset_m_for_ic = current_flute_data_instance.data.get(FLUTE_PARTS_ORDER[0], {}).get('_calculated_stopper_absolute_position_mm', 0.0) / M_TO_MM_FACTOR
                main_bore_points_for_ic = [
                    [(m["position"] / M_TO_MM_FACTOR) - stopper_offset_m_for_ic, m["diameter"] / (2 * M_TO_MM_FACTOR)]
                    for m in current_flute_data_instance.combined_measurements
                ]
                logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - main_bore_points_for_ic (primeros 2): {main_bore_points_for_ic[:2]}")

                # Preparar holes_charac para ImpedanceComputation (metros y radios)
                # Usar side_holes_m_data_raw_original y actualizar la chimenea de la embocadura.
                holes_charac_optimized_for_ic = copy.deepcopy(side_holes_m_data_raw_original)
                embouchure_updated_in_holes_charac = False
                for hole_entry in holes_charac_optimized_for_ic[1:]: # Saltar encabezado
                    if str(hole_entry[0]).lower() == EMBOUCHURE_HOLE_LABEL:
                        hole_entry[2] = optimized_value_m # Actualizar la altura de la chimenea (índice 2) en metros
                        embouchure_updated_in_holes_charac = True
                        logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Chimenea de embocadura en holes_charac_optimized_for_ic actualizada a {optimized_value_m:.5f}m.")
                        break
                if not embouchure_updated_in_holes_charac:
                    logger.error(f"CHECKPOINT OPTIM: Nota {note_name} - No se pudo actualizar la chimenea de la embocadura en holes_charac_optimized_for_ic.")
                logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - holes_charac_optimized_for_ic (primeros 3): {holes_charac_optimized_for_ic[:3]}")

                # The other physics parameters should be from the consistent optim_tool.instru_physics state
                final_optimized_ic_for_storage = ImpedanceComputation(
                    freq_simu_admittance, # Full range used for admittance plots (Positional)
                    main_bore_points_for_ic,                  # Positional
                    holes_charac_optimized_for_ic,            # Positional
                    fing_chart_ow_from_json_original,         # Positional
                    player=optim_tool.instru_physics.player,  # Keyword
                    note=note_name,                           # Keyword
                    temperature=optim_tool.instru_physics.temperature,
                    source_location=optim_tool.instru_physics.source_location,
                    interp=False # Typically False for stored analysis objects for detailed analysis
                )
                logger.info(f"CHECKPOINT OPTIM: Nota {note_name} - ImpedanceComputation final creado exitosamente para almacenamiento.")
                # Optional: Verify antiresonances from this stored IC to ensure it's sensible
                try:
                    ar_check = list(final_optimized_ic_for_storage.antiresonance_frequencies(k=2, display_warning=False))
                    logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - AR del IC recién creado para almacenamiento: {ar_check[:2]}")
                except Exception as e_ar_stored_check:
                    logger.warning(f"CHECKPOINT OPTIM: Nota {note_name} - No se pudieron verificar las AR del IC recién creado: {e_ar_stored_check}")

            except Exception as e_ic_final_create:
                logger.error(f"CHECKPOINT OPTIM: Nota {note_name} - Falló la creación del ImpedanceComputation final para almacenamiento: {e_ic_final_create}", exc_info=True)
                final_optimized_ic_for_storage = None

            optimized_impedance_computations_per_note[note_name] = final_optimized_ic_for_storage
            if final_optimized_ic_for_storage is None:
                 logger.error(f"CHECKPOINT OPTIM: Nota {note_name} - Final: optimized_impedance_computations_per_note[{note_name}] es None.")
                 # If IC creation failed, clear the P/F data for this note as it might be inconsistent
                 pressure_flow_data_per_note[note_name] = {}
            freqs_for_pf_calc = []
            if target_freq > 0: freqs_for_pf_calc.append(target_freq)
            if target_freq > 0: freqs_for_pf_calc.append(2 * target_freq)
            if freqs_for_pf_calc:
                optim_tool.update_frequencies_and_mesh(np.array(freqs_for_pf_calc))
                optim_tool.solve(interp=True, interp_grid=1e-3)
                if hasattr(optim_tool, 'x_interp') and optim_tool.x_interp is not None and \
                   hasattr(optim_tool, 'pressure') and optim_tool.pressure is not None and \
                   hasattr(optim_tool, 'flow') and optim_tool.flow is not None and \
                   hasattr(optim_tool, 'frequencies') and optim_tool.frequencies is not None:
                    pressure_flow_data_per_note[note_name] = {
                        'x_coords': optim_tool.x_interp.copy(),
                        'pressure_modes': optim_tool.pressure.copy(),
                        'flow_modes': optim_tool.flow.copy(),
                        'frequencies': optim_tool.frequencies.copy()
                    }
                    logger.info(f"  Datos de presión/flujo precalculados y guardados para {note_name} desde optim_tool.")
                else:
                    logger.warning(f"  Atributos de P/F NO encontrados o son None en optim_tool para {note_name} después de solve().")
                    pressure_flow_data_per_note[note_name] = {}
            else:
                logger.info(f"  No hay frecuencias válidas para calcular P/F para {note_name}.")
                pressure_flow_data_per_note[note_name] = {}

            notes_actually_optimized.append(note_name)
            logger.debug(f"CHECKPOINT OPTIM: Nota {note_name} - Fin del bloque try de optimización.")
        except Exception as e_opt_note:
            notes_actually_optimized.append(note_name)
            logger.error(f"Error durante la optimización para nota {note_name}: {e_opt_note}", exc_info=True)
            logger.error(f"CHECKPOINT OPTIM: Nota {note_name} - Excepción durante la optimización o post-proceso: {e_opt_note}", exc_info=True)
            optimized_chimney_heights_mm[note_name] = np.nan
            initial_admittance_data[note_name] = (np.array([]), np.array([]))
            optimized_admittance_data[note_name] = (np.array([]), np.array([]))
            pressure_flow_data_per_note[note_name] = {}
            optimized_impedance_computations_per_note[note_name] = None # Asegurar que es None en caso de error

    logger.info("\n--- Resumen de Alturas de Chimenea Optimizadas (mm) ---")
    for note, height in optimized_chimney_heights_mm.items():
        status = f"{height:.3f} mm" if isinstance(height, (int, float)) and not np.isnan(height) else "Error en optimización"
        logger.info(f"  {note}: {status}")

    return optimized_chimney_heights_mm, initial_admittance_data, optimized_admittance_data, flute_model_name, target_frequencies_hz_map, diapason_a4_hz_gui, optimized_impedance_computations_per_note, pressure_flow_data_per_note, notes_actually_optimized, initial_acoustic_analysis_dict # type: ignore

if __name__ == "__main__":
    # Ejemplo de uso:
    # Necesitas un directorio de flauta válido para probar.
    # test_flute_directory_for_bore = "/ruta/a/tu/directorio/de/flauta_json"
    # if Path(test_flute_directory_for_bore).is_dir():
    #     optimized_heights, initial_adm_data, optimized_adm_data, model_name_res, target_freqs_res, diapason_res, _, pf_data, _ = optimize_flute_from_json_full(
    #         flute_dir_path_str=test_flute_directory_for_bore,
    #         diapason_a4_hz_gui=DIAPASON_A4_HZ,
    #         target_temp_c_gui=TARGET_TEMPERATURE_C
    #     )
    #     if optimized_adm_data:
    #         plot_optimized_admittances(optimized_adm_data, model_name_res, target_freqs_res, diapason_res)
    #         plt.show()

    #     if pf_data:
    #         for note, data in pf_data.items():
    #             if data and 'x_coords' in data:
    #                 print(f"\nDatos P/F para nota {note}:")
    #                 print(f"  X Coords shape: {data['x_coords'].shape}")
    #                 print(f"  Pressure Modes shape: {data['pressure_modes'].shape}")
    #                 print(f"  Flow Modes shape: {data['flow_modes'].shape}")
    #                 print(f"  Frequencies: {data['frequencies']}")
    #             else:
    #                 print(f"\nNo hay datos P/F válidos para nota {note}")
    # else:
    #     print(f"Directorio de prueba no encontrado: {test_flute_directory_for_bore}")
    logger.info("Ejecución de prueba de optimize_flute_from_json.py completada (comentada).")
