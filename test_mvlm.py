from pathlib import Path
import numpy as np
import mvlm

file = Path("testing/sculptor_neutral.obj")

dm = mvlm.pipeline.create_pipeline(
    "face_alignment",
    render_image_stack=True,
    render_image_folder="testing/mvlm/render_image_stack",
)
landmarks = dm.predict_one_file(file)
np.save("testing/sculptor/landmarks.npy", landmarks)

mvlm.utils.VTKViewer(file.as_posix(), landmarks)

print(landmarks)
