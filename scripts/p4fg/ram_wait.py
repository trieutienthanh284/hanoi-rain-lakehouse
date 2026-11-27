"""Wait until free RAM >= 0.5 GB (TASK P4F/P4G RAM rule) before a unit-test file; poll every 10 s for at most 5 min.
Exit 0 when the condition holds, 3 when free RAM stayed below the limit for 5 minutes (stop condition)."""
import sys
import time

import psutil

LIMIT_GB, MAX_S, STEP_S = 0.5, 300, 10

t0 = time.time()
while True:
    free = psutil.virtual_memory().available / 2**30
    if free >= LIMIT_GB:
        print(f"RAM_OK free_gb={free:.2f} waited_s={time.time() - t0:.0f}")
        sys.exit(0)
    if time.time() - t0 >= MAX_S:
        print(f"RAM_STOP free_gb={free:.2f} below {LIMIT_GB} GB for {MAX_S} s")
        sys.exit(3)
    time.sleep(STEP_S)
