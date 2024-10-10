import os,sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.insert(0, src_path)
from Src.economy import EconomyModel
import cProfile
import pstats
import io
import logging
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_model(steps, timeout=300):  # 5 minutes timeout
    model = EconomyModel(num_workers=30, num_firm1=0, num_firm2=5, mode='decentralised')
    for i in range(steps):
        model.step()
    return model

def profile_model_run(steps):
    pr = cProfile.Profile()
    pr.enable()

    # Run the model
    model = run_model(steps)

    pr.disable()

    # Print profiling results
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    return model

if __name__ == "__main__":
    steps = 20  # Number of steps to run
    logging.info(f"Starting model run with profiling for {steps} steps...")

    model = profile_model_run(steps)

    if model:
        logging.info("Model completed successfully.")
    else:
        logging.error("Model did not complete successfully.")
