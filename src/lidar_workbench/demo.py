import json
from .synthetic import scene
from .geometry import obstacle_pipeline
from .landing import landing_candidate

def main():
    points=scene()
    obstacles=obstacle_pipeline(points)
    print(json.dumps({"data":"synthetic","input_points":len(points),
        "clusters":obstacles["clusters"],"landing_candidate":landing_candidate(points)},indent=2))
if __name__=="__main__":main()
