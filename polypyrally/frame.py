class Frame:
    """Everything the renderer needs for one frame."""

    def __init__(self):
        self.time = 0.0
        self.view = None
        self.proj = None
        self.cam_pos = (0.0, 0.0, 0.0)
        self.forward = (0.0, 0.0, -1.0)
        self.focus = (0.0, 0.0, 0.0)
        self.terrain = []         # (GLMesh, centre x, centre z, radius)
        self.inst_main = []       # (GLMesh, GLInstances) drawn in the colour pass
        self.inst_shadow = []     # (GLMesh, GLInstances) drawn in the shadow passes
        self.models = []          # (GLMesh, 4x4 model matrix, gloss)
        self.ui = None
        self.particles = None     # float32 (N, 9) vertices: pos3 uv2 rgba4
