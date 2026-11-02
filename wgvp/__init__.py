"""wgvp - WearGait video-to-pose pipeline (run-it-yourself package for a data partner).

Stages (CPU unless noted):
  analyze   one decode pass per video: multi-person 2D pose tracking (YOLO-pose + ByteTrack) and a coarse
            brightness grid for the sync light
  select    pick the subject among the tracked people (IMU cross-correlation / walkway ROI / operator pick)
  sync      video frame of IMU t=0 (light flash / IMU cross-correlation / annotation sheet)
  cut       trim + bbox track for the lifter (frame mapping kept)
  gvhmr     [GPU] SMPL-X parameters per frame for the subject (GVHMR, static camera, no render)
  reconcile two camera angles against each other on the IMU timeline
  qc        QC records, skeleton-only thumbnails, validation bundle, privacy check

Only pose-level derivatives and scalar QC leave the partner's site; see README.md ("Privacy").
"""
__version__ = "0.1.0"
