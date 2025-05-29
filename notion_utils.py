import logging
from typing import Tuple, Optional, Dict, Any, List
from notion_client import Client
import requests

logger = logging.getLogger(__name__)

def get_json_files_from_notion(notion_token: str, database_id: str, flute_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Recupera los datos JSON relacionados con una flauta específica desde Notion.

    Args:
        notion_token (str): Token de la API de Notion.
        database_id (str): ID de la base de datos en Notion.
        flute_name (str): Nombre de la flauta a buscar.

    Returns:
        tuple: Diccionarios con los datos de cada parte (headjoint, left, right, foot) o None en caso de error.
    """
    try:
        client = Client(auth=notion_token)
        # Buscar la flauta por nombre
        results = client.databases.query(
            database_id=database_id,
            filter={
                "property": "Name",
                "title": {"contains": flute_name}
            }
        )

        if not results.get("results"):
            logger.error("No se encontró una flauta con el nombre: %s", flute_name)
            return None, None, None, None

        # Recuperar la primera página de resultados
        page = results["results"][0]
        related_ids = {
            "headjoint": page["properties"]["headjoint"]["relation"][0]["id"],
            "left": page["properties"]["left"]["relation"][0]["id"],
            "right": page["properties"]["right"]["relation"][0]["id"],
            "foot": page["properties"]["foot"]["relation"][0]["id"]
        }

        # Recuperar datos JSON desde las páginas relacionadas
        json_data = {}
        for part, related_id in related_ids.items():
            json_data[part] = download_related_page_json(client, related_id)

        return json_data.get("headjoint"), json_data.get("left"), json_data.get("right"), json_data.get("foot")

    except Exception as e:
        logger.exception("Error al obtener datos JSON desde Notion: %s", e)
        return None, None, None, None

def download_related_page_json(client: Client, page_id: str) -> Dict[str, Any]:
    """
    Descarga el contenido JSON desde una página relacionada en Notion.

    Args:
        client (Client): Cliente de la API de Notion.
        page_id (str): ID de la página relacionada.

    Returns:
        dict: Contenido JSON de la página relacionada, o un diccionario vacío en caso de error.
    """
    try:
        page = client.pages.retrieve(page_id=page_id)
        # Extraer la URL del archivo JSON desde '_external_object_url'
        file_url = page["properties"]["_external_object_url"]["url"]

        # Convertir la URL para descargar directamente desde Google Drive
        if "drive.google.com" in file_url:
            file_id = file_url.split("/d/")[1].split("/")[0]
            file_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(file_url)
        response.raise_for_status()
        return response.json()
    except KeyError as e:
        logger.error("La página no contiene la propiedad esperada '_external_object_url': %s", e)
        return {}
    except requests.exceptions.RequestException as e:
        logger.error("Error al descargar el archivo JSON: %s", e)
        return {}

def get_flute_names_from_notion(notion_token: str, database_id: str) -> List[str]:
    """
    Obtiene una lista de nombres de flautas desde una base de datos de Notion.

    Args:
        notion_token (str): Token de integración para acceder a la API de Notion.
        database_id (str): ID de la base de datos de Notion.

    Returns:
        list: Lista de nombres de flautas extraídos de la base de datos.
    """
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    url = f"https://api.notion.com/v1/databases/{database_id}/query"

    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        flute_names = []
        for result in data.get("results", []):
            properties = result.get("properties", {})
            name_property = properties.get("Name", {})
            if "title" in name_property:
                name = "".join([text.get("plain_text", "") for text in name_property["title"]])
                if name:
                    flute_names.append(name)
        return flute_names

    except requests.exceptions.RequestException as e:
        logger.error("Error al comunicarse con la API de Notion: %s", e)
        return []
    except Exception as e:
        logger.exception("Error inesperado al procesar los datos de Notion: %s", e)
        return []