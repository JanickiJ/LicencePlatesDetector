# Licence palates detector

Project consist of a few executable files.

* main - hosts webapp that visualize camera view and predictions. Run in main dir by `python main.py`
* detection_contours - uses contours detection as engine. Depending on chosen mode. If debug = True app shows camera Run in main dir by `python -m src.experiments.detection_contours`
  view with rectangles that contains detected object also prints predictions. If debug = False, only prints predictions
* detection_text_fields - uses text fields detection as engine. As above, works in to modes. Run in main dir by `python -m src.experiments.detection_text_fields`
* comparison - compare two detection engines using static pictures, measure time of detection, visualise results. Run in main dir by `python -m src.experiments.comparison`