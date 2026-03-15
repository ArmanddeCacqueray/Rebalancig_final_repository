Final archive associated with the rebalancing collaboration project between Inria and Smovengo. 
It includes the full 2-year dataset and pratical implementation of our core algorithms.
====================
====================
DATABASE: full_db.nc (Xarrray 300MB hourly database) 
====================
====================
PIPELINE goes from a to e:
====================
====================
a- process PAST stock and flows (under censorship context) to compute valid demand estimation
====================
b- process PAST demand estimation to reconstruct stock evolution in 4 fictive rebalancing scenarios (so called envelopes)
 [global_MIN, global_MAX] scenarios,
 [balance_min, balance_max] scenarios
====================
c-process PAST 4 scenarios history to forecast FUTURE 4 scenarios
(up to one week forecast)
====================
d-build a simulator upon forecasted FUTURE  4 scenarios to evaluate ANY
rebalancing policy and:
classify stations among priority, buffer, neutral
compute policy frontiers for admissible strategies in term of (feasibility + metropolis score  metric)
====================
e-build a milp optimization solver upon  FUTURE computed admissibility  frontiers, aiming  to select an efficient (transport cost) rebalancing TONIGHT (but though as part of a weekly scaled plan and thus we solve an extended multidays problem) plan among admissible ones
====================
====================
BENCHMARK fully reproducible benchmark for forecasting model
====================
====================


