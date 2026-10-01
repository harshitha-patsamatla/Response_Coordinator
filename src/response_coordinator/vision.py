"""YOLO object detection and EXIF GPS extraction for uploaded images."""

from __future__ import annotations

import os
from pathlib import Path


VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "bus", "truck"}
FIRE_CLASSES = {"fire", "smoke", "flame"}
CRASH_CLASSES = {"crashed vehicle", "damaged vehicle", "vehicle damage"}


def read_exif_gps(image_path: str | Path) -> tuple[float, float] | None:
    from PIL import ExifTags, Image

    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            latitude = _gps_decimal(gps_ifd.get(2), gps_ifd.get(1))
            longitude = _gps_decimal(gps_ifd.get(4), gps_ifd.get(3))
            if latitude is not None and longitude is not None:
                return latitude, longitude
    except (OSError, AttributeError, KeyError, TypeError, ZeroDivisionError):
        return None
    return None


def analyze_image(image_path: str | Path) -> dict:
    from ultralytics import YOLO

    custom_weights = os.getenv("COORDINATOR_VISION_WEIGHTS")
    weights = custom_weights or "yolov8n.pt"
    model = YOLO(weights)
    results = model(str(image_path), verbose=False)
    names = model.names
    detections: dict[str, int] = {}
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        label = str(names[class_id]).casefold()
        detections[label] = detections.get(label, 0) + 1
    detected_labels = set(detections)
    return {
        "people_count": detections.get("person", 0),
        "vehicle_count": sum(detections.get(name, 0) for name in VEHICLE_CLASSES),
        "fire_smoke_count": sum(detections.get(name, 0) for name in FIRE_CLASSES),
        "crash_damage_count": sum(detections.get(name, 0) for name in CRASH_CLASSES),
        "fire_smoke_available": bool(detected_labels & FIRE_CLASSES) or bool(custom_weights),
        "crash_damage_available": bool(detected_labels & CRASH_CLASSES) or bool(custom_weights),
        "detected_objects": detections,
        "weights": weights,
        "gps": read_exif_gps(image_path),
    }


def _gps_decimal(value, reference) -> float | None:
    if value is None or reference is None:
        return None
    degrees, minutes, seconds = (float(part) for part in value)
    decimal = degrees + minutes / 60 + seconds / 3600
    return -decimal if str(reference).upper() in {"S", "W"} else decimal