from pathlib import Path
import numpy as np
import mvlm

file = Path("testing/flame_mesh.obj")

dm = mvlm.pipeline.create_pipeline(
    "face_alignment",
    render_image_stack=True,
    render_image_folder=Path("testing/mvlm/render_image_stack"),
)
landmarks = dm.predict_one_file(file)
np.save("testing/sculptor/landmarks.npy", landmarks)

print(landmarks.shape)

mvlm.utils.VTKViewer(file.as_posix(), landmarks)
