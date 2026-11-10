# Public scientific invariant checks

From the repository root, install the package and run:

```sh
python -m pip install -e .
python -m unittest discover -s tests -p test_invariants.py -v
```

The eleven checks use synthetic arrays and temporary CSV/checkpoint files. They read no study data, contact no remote service and compute no study score. Two tiny fits use two clients, two rounds, one local epoch and one CPU thread each. Test prediction values are synthetic fixtures and are not benchmark results.

The suite checks source hashes and calendars, missing-value rejection, explicit monthly availability, December label embargoes and chronology, future-data independence of training statistics, actual count-weighted parameter aggregation, FedProx with zero proximal strength, label-free inference, checkpoint reload and dropout evaluation, effective climate-conditioned attention, disabled-context invariance and effective fixed-anchor ablations. These checks protect contracts; they do not establish empirical skill, historical source vintages, cryptographic privacy or correctness of an external data extraction.

The nonzero CAA anchor is separate from a selected coefficient candidate. A selected boundary candidate may use only one term; removal of that sole term is rejected. The public ablation parent keeps all three terms active.
