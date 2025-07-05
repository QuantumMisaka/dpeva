import os
import glob

max_subjob_per_job = 50
sub_count = 0
main_count = 0

for dir in sorted(glob.glob("./*/")):
    if ("CONVERGED" in dir) or ("N_" in dir):
        continue
    else:
        jobdir = f"N_{max_subjob_per_job}_{main_count}"
        if not os.path.exists(jobdir):
                os.mkdir(jobdir)
        os.system(f"mv {dir} {jobdir}")
        sub_count += 1
        if sub_count >= max_subjob_per_job:
                main_count += 1
                sub_count = 0
print("DONE!")