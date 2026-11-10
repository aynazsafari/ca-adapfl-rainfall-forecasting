# Release-candidate validation

Validated locally on the recorded Windows/Python runtime. Source packaging
adds tools and docs without changing606 byte-preserved scientific/source files.
The paper evidence contains513 completed fit bundles across four studies.
No new full empirical benchmark training was performed during packaging.

| Check | Outcome |
|---|---|
| Primary independent metrics/cost/paired uncertainty |PASS;96 labels,230400 forecasts,186000 draws|
| Source controls independent arithmetic/cost |PASS_OUTPUT_ARITHMETIC;36 labels,83100 forecasts,108000 draws|
| Public source admission/scaling |Solar/GLDAS both arms and IMD REFERENCE checked; IMD ADDED withheld|
| Capacity independent metrics/uncertainty |PASS;46 labels,110400 rows,294000 draws|
| Sharing independent QA |PASS;75331 assertions,415487 numeric scalar checks|
| Primary saved-state replay |230400 rows; neural/baseline exact; classical/router max1.14e-13mm, stated tolerance1e-10mm|
| Capacity replay/carries |110400 exact:48000 new inference,24000 aliases/projections,38400 authenticated archived carries|
| Public source replay |57600 exact solar/GLDAS rows; explicit IMD PARTIAL_UNAVAILABLE|
| Complete local source replay |83100 exact rows using local IMD inputs|
| Sharing state replay |62400 neural forecasts exact;2400 no-fit climatology rows checked by arithmetic QA|
| Native IMD reconstruction |44 native hashes;8 restored source/prepared paths byte-identical; no public input redistribution|
| Packaging/unit contracts |29 passed; corruption, traversal, overwrite and fresh-training guards|
| Scientific implementation invariants |11 passed; chronology, scaling, active components, deterministic state reload|
| Source packaging synthetic contracts |27 passed;2 symlink-creation cases unavailable due to Windows privilege; guard code reviewed|
| Executed notebook |7/7 cells passed;96-label metric discrepancy at most1.42e-14; HTML/plot visually inspected|
| Publication exports |31 tables+9 companions;514 displayed result values agree;45-item evidence map|
| Figure QA |10 numerical redraws+4 method redraws inspected; original13 main figure images retained|

`metadata/validation_summary.json` records the receipt SHA256 identities and
scientific boundaries. Full result completion/audit receipts remain in result
assets; fresh local receipts are under `work/`. IMD native/imported values remain
local and outside the deposited source whitelist. Four complete arithmetic
scopes pass, but public IMD-added source/state reconstruction is incomplete
until local acquisition/import. The historical same-month IMD descriptive pairs
file is not rebuilt by the importer and is not required for current comparisons.

The recorded runtime matches on this host. Installing nominally matching
dependencies on a different host is not proof of exact compiler/Torch parity.
Hosted CI has not run. Results and times concern a one-host retrospective
simulation, not a fresh unseen evaluation or institutional deployment.
