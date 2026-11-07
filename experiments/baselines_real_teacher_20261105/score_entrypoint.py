"""Run the frozen scorer with its missing console-only json import supplied.

The original scorer writes every result and chart before its final json.dumps
print. Injecting this standard-library module fixes that print while retaining
the exact frozen scoring source, thresholds, and metrics.
"""
import json
import runpy
from common import RUN

if __name__ == '__main__':
    runpy.run_path(str(RUN / 'score_results.py'), init_globals={'json': json}, run_name='__main__')
