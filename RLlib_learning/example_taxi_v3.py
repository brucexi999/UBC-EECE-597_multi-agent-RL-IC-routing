import ray
import os
import ray.rllib.algorithms.ppo as ppo
import shutil

# Start Ray running in the background
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Show Ray dashboard
#print("Dashboard URL: http://{}".format(ray.get_webui_url())) 


# Configure a file location for checkpoints
CHECKPOINT_ROOT = "tmp/ppo/taxi"
shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

ray_results = os.getenv("HOME") + "/ray_results/"
shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

