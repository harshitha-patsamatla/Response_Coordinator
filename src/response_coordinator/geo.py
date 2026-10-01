"""Swappable OSM geocoding, nearby-service search, and ETA routing."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import requests

from .config import ENABLE_OSM_LOOKUPS, NOMINATIM_USER_AGENT
from .schemas import AssignedTeam, Location


class OpenStreetMapProvider:
    def __init__(self, mock_path: Path, landmarks_path: Path) -> None:
        self.mock_data = json.loads(mock_path.read_text(encoding="utf-8"))
        self.landmarks = json.loads(landmarks_path.read_text(encoding="utf-8"))
        self.enabled = ENABLE_OSM_LOOKUPS

    def resolve_location(
        self,
        text: str | None,
        source: str | None,
        landmark_coordinates: dict | None = None,
        image_gps: tuple[float, float] | None = None,
        reported_gps: tuple[float, float] | None = None,
    ) -> Location:
        if text and source == "explicit" and self.enabled:
            result = self._nominatim_geocode(text)
            if result:
                return Location(text=text, lat=result[0], lon=result[1], confidence=0.78, source="nominatim")
        if landmark_coordinates:
            return Location(
                text=text,
                lat=float(landmark_coordinates["lat"]),
                lon=float(landmark_coordinates["lon"]),
                confidence=0.85,
                source="local_landmark",
            )
        if reported_gps:
            return Location(
                text=text or "Reported GPS coordinates",
                lat=reported_gps[0],
                lon=reported_gps[1],
                confidence=0.95,
                source="report_gps",
            )
        if text and source == "landmark":
            for name, coordinates in self.landmarks.items():
                if name.casefold() == text.casefold():
                    return Location(
                        text=name,
                        lat=float(coordinates["lat"]),
                        lon=float(coordinates["lon"]),
                        confidence=0.85,
                        source="local_landmark",
                    )
        if image_gps:
            return Location(
                text=text or "Image GPS location",
                lat=image_gps[0],
                lon=image_gps[1],
                confidence=0.65,
                source="image_exif_gps",
            )
        return Location(text=text, source=None)

    def find_nearby_facilities(self, lat: float, lon: float, limit: int = 8) -> list[dict[str, Any]]:
        facilities = self._overpass_facilities(lat, lon) if self.enabled else []
        facilities.extend(self.mock_data["facilities"])
        return self._rank_by_eta(facilities, lat, lon, limit)

    def nearest_team(self, lat: float, lon: float, incident_types: list[str] | None = None) -> AssignedTeam:
        teams = self._overpass_facilities(lat, lon, response_teams=True) if self.enabled else []
        preferred = "fire_station" if "Fire" in (incident_types or []) else "police" if "Crime" in (incident_types or []) else "ambulance"
        teams = [team for team in teams if team["type"] == preferred]
        if not teams:
            teams = [team for team in self.mock_data["response_teams"] if team["type"] == preferred]
        if not teams:
            teams = list(self.mock_data["response_teams"])
        ranked = self._rank_by_eta(teams, lat, lon, 1)
        if not ranked:
            return AssignedTeam(source="unavailable")
        best = ranked[0]
        return AssignedTeam(
            name=best["name"],
            eta_min=best["eta_min"],
            source=f"{best['source']} · {best['route_source']}",
        )

    def _nominatim_geocode(self, query: str) -> tuple[float, float] | None:
        try:
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "jsonv2", "limit": 1},
                headers={"User-Agent": NOMINATIM_USER_AGENT},
                timeout=3,
            )
            response.raise_for_status()
            results = response.json()
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
        except (requests.RequestException, ValueError, KeyError):
            return None
        return None

    def _overpass_facilities(self, lat: float, lon: float, response_teams: bool = False) -> list[dict]:
        tags = ["fire_station", "police"] if response_teams else ["hospital", "clinic", "police", "fire_station"]
        clauses = "".join(
            f'node(around:15000,{lat},{lon})[amenity={tag}];way(around:15000,{lat},{lon})[amenity={tag}];'
            for tag in tags
        )
        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": f"[out:json][timeout:5];({clauses});out center tags;"},
                headers={"User-Agent": NOMINATIM_USER_AGENT},
                timeout=6,
            )
            response.raise_for_status()
            items = []
            for element in response.json().get("elements", []):
                tags_found = element.get("tags", {})
                element_lat = element.get("lat", element.get("center", {}).get("lat"))
                element_lon = element.get("lon", element.get("center", {}).get("lon"))
                if element_lat is None or element_lon is None:
                    continue
                amenity = tags_found.get("amenity", "response")
                items.append({
                    "name": tags_found.get("name", f"Nearby {amenity.replace('_', ' ')}"),
                    "type": "fire_station" if amenity == "fire_station" else "police" if amenity == "police" else "hospital",
                    "lat": float(element_lat),
                    "lon": float(element_lon),
                    "source": "openstreetmap",
                })
            return items
        except (requests.RequestException, ValueError):
            return []

    def _rank_by_eta(self, facilities: list[dict], lat: float, lon: float, limit: int) -> list[dict]:
        ranked = []
        candidates = sorted(
            facilities,
            key=lambda item: _haversine(lat, lon, item["lat"], item["lon"]),
        )[:4]
        for facility in candidates:
            eta = self._osrm_eta(lat, lon, facility["lat"], facility["lon"])
            if eta is None:
                distance_km = _haversine(lat, lon, facility["lat"], facility["lon"])
                eta = max(1, round(distance_km * 1.4))
                route_source = "mock-estimate"
            else:
                route_source = "osrm"
            ranked.append({**facility, "eta_min": eta, "route_source": route_source})
        return sorted(ranked, key=lambda item: item["eta_min"])[:limit]

    def _osrm_eta(self, lat1: float, lon1: float, lat2: float, lon2: float) -> int | None:
        if not self.enabled:
            return None
        try:
            response = requests.get(
                f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}",
                params={"overview": "false"},
                timeout=2.5,
            )
            response.raise_for_status()
            seconds = response.json()["routes"][0]["duration"]
            return max(1, math.ceil(seconds / 60))
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None


class GoogleMapsProvider(OpenStreetMapProvider):
    """Google Maps implementation of the same geocoding/search/routing surface."""

    def __init__(self, mock_path: Path, landmarks_path: Path, api_key: str | None = None) -> None:
        super().__init__(mock_path, landmarks_path)
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.enabled = self.enabled and bool(self.api_key)

    def _nominatim_geocode(self, query: str) -> tuple[float, float] | None:
        if not self.enabled:
            return None
        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": query, "key": self.api_key},
                timeout=3,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                point = results[0]["geometry"]["location"]
                return float(point["lat"]), float(point["lng"])
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None
        return None

    def _overpass_facilities(self, lat: float, lon: float, response_teams: bool = False) -> list[dict]:
        if not self.enabled:
            return []
        place_types = ["fire_station", "police"] if response_teams else ["hospital", "police", "fire_station"]
        facilities = []
        try:
            for place_type in place_types:
                response = requests.get(
                    "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                    params={
                        "location": f"{lat},{lon}",
                        "radius": 15000,
                        "type": place_type,
                        "key": self.api_key,
                    },
                    timeout=3,
                )
                response.raise_for_status()
                for place in response.json().get("results", []):
                    point = place["geometry"]["location"]
                    facilities.append({
                        "name": place.get("name", f"Nearby {place_type.replace('_', ' ')}"),
                        "type": place_type,
                        "lat": float(point["lat"]),
                        "lon": float(point["lng"]),
                        "source": "google_maps",
                    })
        except (requests.RequestException, ValueError, KeyError):
            return facilities
        return facilities

    def _osrm_eta(self, lat1: float, lon1: float, lat2: float, lon2: float) -> int | None:
        if not self.enabled:
            return None
        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params={
                    "origin": f"{lat1},{lon1}",
                    "destination": f"{lat2},{lon2}",
                    "mode": "driving",
                    "key": self.api_key,
                },
                timeout=3,
            )
            response.raise_for_status()
            seconds = response.json()["routes"][0]["legs"][0]["duration"]["value"]
            return max(1, math.ceil(seconds / 60))
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None


def create_geo_provider(mock_path: Path, landmarks_path: Path):
    """Select OSM (default) or Google Maps using MAP_PROVIDER."""
    if os.getenv("MAP_PROVIDER", "osm").casefold() == "google":
        return GoogleMapsProvider(mock_path, landmarks_path)
    return OpenStreetMapProvider(mock_path, landmarks_path)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(delta_lon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))