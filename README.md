# Run
Particle Filter tracking
```
python particle_run.py --input_video <video_name> --out_file <out_video_path> 
```
By default, the software allows user to select point of interest via clicking on the scene. If user has an annotation file of selected points in json format, add `--ann_file <annotation of selected points>`.

Sift tracking
```
python sift_tracking_run.py --input_video <video_name>
```