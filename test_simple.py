from openwind import Player
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    p = Player()
    if hasattr(p, 'labels'):
        logger.info(f"Prueba mínima: Player tiene 'labels': {p.labels}")
    else:
        logger.error("CRÍTICO: Prueba mínima: Player NO TIENE 'labels'.")
except Exception as e:
    logger.error(f"Error creando Player en prueba mínima: {e}", exc_info=True)
