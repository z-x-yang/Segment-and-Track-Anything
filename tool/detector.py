class Detector:
    def __init__(self, device):
        print("GroundingDINO disabled")

    def run_grounding(self, *args, **kwargs):
        raise RuntimeError(
            "Grounding detector is disabled in this build."
        )