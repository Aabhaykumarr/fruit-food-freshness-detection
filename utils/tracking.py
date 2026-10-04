"""
Object tracking and temporal cooldown utility.
Tracks detected objects across frames using bounding-box overlap (IoU)
and ensures we don't call the LLM 30 times per second for the same object.
"""

import time
from typing import List, Dict, Any, Optional


def compute_iou(boxA, boxB) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea)


class TrackedObject:
    """Represents a single tracked food/fruit item across frames."""

    def __init__(self, track_id: int, class_name: str, bbox: List[int], confidence: float):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_analyzed = 0.0  # timestamp of last LLM analysis
        self.latest_analysis: Optional[Dict[str, Any]] = None

    def update(self, bbox: List[int], confidence: float):
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()

    def should_analyze(self, cooldown_seconds: float) -> bool:
        """Returns True if this object needs an LLM analysis (first time or cooldown passed)."""
        return (time.time() - self.last_analyzed) >= cooldown_seconds

    def record_analysis(self, analysis_result: Dict[str, Any]):
        self.last_analyzed = time.time()
        self.latest_analysis = analysis_result


class ObjectTracker:
    """
    Simple IoU-based multi-object tracker.
    Maintains persistent IDs for detected items across frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost_seconds: float = 3.0):
        self.iou_threshold = iou_threshold
        self.max_lost_seconds = max_lost_seconds
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Match incoming YOLO detections to existing tracks or create new ones.
        Returns the active TrackedObject instances.
        """
        now = time.time()

        # Remove stale tracks
        lost_ids = [
            tid for tid, track in self.tracks.items()
            if (now - track.last_seen) > self.max_lost_seconds
        ]
        for tid in lost_ids:
            del self.tracks[tid]

        matched_tracks = []
        unmatched_detections = list(detections)

        # Match detections to existing tracks by class + IoU
        for tid, track in list(self.tracks.items()):
            best_iou = 0.0
            best_det_idx = -1

            for idx, det in enumerate(unmatched_detections):
                if det["class_name"] == track.class_name:
                    iou = compute_iou(track.bbox, det["bbox"])
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_det_idx = idx

            if best_det_idx >= 0:
                det = unmatched_detections.pop(best_det_idx)
                track.update(det["bbox"], det["confidence"])
                matched_tracks.append(track)

        # Create new tracks for unmatched detections
        for det in unmatched_detections:
            track = TrackedObject(
                track_id=self._next_id,
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
            )
            self.tracks[self._next_id] = track
            self._next_id += 1
            matched_tracks.append(track)

        return matched_tracks
