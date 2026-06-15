from typing import Any, Dict, List

import cv2


def get_track_color(track_id: str):
    try:
        value = int(track_id)
    except ValueError:
        value = sum(ord(c) for c in str(track_id))

    # Deterministic color from ID.
    # Example for ID=5:
    #   b=(29*5)%255=145, g=(17*5)%255=85, r=(37*5)%255=185.
    b = (29 * value) % 255
    g = (17 * value) % 255
    r = (37 * value) % 255

    return int(b), int(g), int(r)


def draw_tracks(frame, tracks: List[Dict[str, Any]]):
    output_frame = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = track["bbox_xyxy"]
        track_id = track["track_id"]
        class_name = track.get("class_name", "vehicle")
        confidence = float(track.get("confidence", 0.0))
        visibility = int(track.get("visibility", 0))

        color = get_track_color(track_id)

        if visibility == 1:
            label = f"ID {track_id} | {class_name} {confidence:.2f}"
        else:
            label = f"ID {track_id} | predicted"

        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

        # Keep label text inside the frame.
        # Example: y1=10 -> y1-8=2, so text_y becomes 20.
        text_y = max(20, y1 - 8)

        cv2.rectangle(
            output_frame,
            (x1, text_y - 18),
            (min(x1 + 260, output_frame.shape[1] - 1), text_y + 4),
            color,
            -1
        )

        cv2.putText(
            output_frame,
            label,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return output_frame


def draw_hud(frame, items: Dict[str, Any]):
    output_frame = frame.copy()

    x = 12
    y = 28
    line_height = 26

    for key, value in items.items():
        text = f"{key}: {value}"

        cv2.putText(
            output_frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        y += line_height

    return output_frame
