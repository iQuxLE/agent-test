import os
import requests
from typing import Tuple, Optional


def get_static_map(latitude: float, longitude: float, zoom: int = 13, 
                  size: Tuple[int, int] = (600, 400),
                  marker_color: str = "red",
                  maptype: str = "satellite") -> Optional[bytes]:
    """
    Fetches a static map image from Google Maps API.
    
    :param latitude: Latitude coordinate
    :param longitude: Longitude coordinate
    :param zoom: Zoom level (1-20)
    :param size: Image size as (width, height) tuple
    :param marker_color: Color of the marker
    :param maptype: Type of map (roadmap, satellite, hybrid, terrain)
    :return: Raw image bytes or None if request failed
    """
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set")
    
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "markers": f"color:{marker_color}|{latitude},{longitude}",
        "maptype": maptype,
        "key": api_key
    }
    print(f"Fetching map with params: {params}")
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.content
    except requests.RequestException as e:
        print(f"Error fetching map: {e}")
        return None