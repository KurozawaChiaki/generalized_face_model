from pathlib import Path
import numpy as np
import mvlm

file = Path("testing/test2.obj")

dm = mvlm.pipeline.create_pipeline("face_alignment")
landmarks = dm.predict_one_file(file)

mvlm.utils.VTKViewer(file.as_posix(), landmarks)

print(landmarks)
