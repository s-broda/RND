import sys; sys.path.insert(0, "python")
import joe
joe.run_mc(n_reps=200, sds=(0.01,), chains=("dense", "sparse", "listed"))
joe.run_mc(n_reps=100, sds=(0.005, 0.02), chains=("dense", "sparse"))
print("MC DONE")
