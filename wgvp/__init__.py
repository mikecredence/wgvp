"""wgvp - WearGait video-to-pose pipeline (run-it-yourself package for a data partner).

Stages (CPU unless noted):
  analyze   one decode pass per video: multi-person 2D pose tracking (YOLO-pose + ByteTrack) and a coarse
            brightness grid for the sync light
  select    pick the subject among the tracked people (IMU cross-correlation / walkway ROI / operator pick)
  sync      video frame of IMU t=0 (light flash / IMU cross-correlation / annotation sheet)
  cut       trim + bbox track for the lifter (frame mapping kept)
  gvhmr     [GPU] SMPL-X parameters per frame for the subject (GVHMR, static camera, no render)
  reconcile two camera angles against each other on the IMU timeline
  merge     one pose per frame from the two views, each joint from the view(s) that can vouch for it (merged_smpl.npz;
            no IMU enters it)
  jointfit  one body fitted to both views' keypoints at once (jointfit_smpl.npz); target.json then names the training
            target of the recording: the fit where it applies and passes its checks, the merge everywhere else
  demo      [optional extras] the demo video of a lifted recording, from the outputs above (work/, holds video frames)
  qc        QC records, skeleton-only thumbnails, validation bundle, privacy check

Only pose-level derivatives and scalar QC leave the partner's site; see README.md ("Privacy").
"""
__version__ = "0.3.0"
