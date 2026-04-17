# Instruction: Refactor Aiyagari (1994) Notebook into Julia Files

## Goal

Refactor the Jupyter notebook `Ch5/aiyagari.ipynb` into modular Julia source files under `Ch5/src/` and a main script `Ch5/aiyagari.jl`, following the structure of the original textbook code at `quant_macro_book/chapter5/Julia/`.

## Context

- **Reference code**: `quant_macro_book/chapter5/Julia/aiyagari_1994.jl` and its helper files (`aiyagari_vfi1.jl`, `aiyagari_vfi2.jl`, `aiyagari_vfi3.jl`, `tauchen.jl`, `aiyagari_transition.jl`)
- **Source notebook**: `Ch5/aiyagari.ipynb` — contains a working implementation of the Aiyagari (1994) model with both stationary equilibrium and transition dynamics
- The notebook has several improvements over the reference code (government budget constraint, larger capital grid, transition endpoint conditions)

## File Structure

Create the following files:

```
Ch5/
├── aiyagari.jl                  # Main script that includes all sources and runs the model
├── INSTRUCTION.md               # This file
├── images/                      # Output directory for figures
└── src/
    ├── tauchen.jl               # Tauchen discretization of AR(1) process
    ├── model.jl                 # Model struct, create_model(), calculate_K0(), calculate_w0()
    ├── grid.jl                  # generate_capital_grid(), translate_capital_grid()
    ├── vfi.jl                   # solve_VFI() — grid-search version
    ├── vfi_continuous.jl        # solve_VFI_continuous() — continuous-control golden section search
    ├── distribution.jl          # derive_stationary_distribution(), calculate_capital_supply()
    ├── equilibrium.jl           # solve_stationary_equilibrium(), derive_capital_supply_curve()
    ├── transition.jl            # Grid-search transition functions and derive_transition()
    └── transition_continuous.jl # Continuous-control backward VFI for transition
```

## Detailed Instructions per File

### `src/tauchen.jl`
- Implement `tauchen(N, rho, sigma, m)` — discretizes AR(1) process using Tauchen's method
- Returns `(Z, Zprob, Zinv)`: grid, transition matrix, stationary distribution
- Uses `Distributions.jl` for the normal CDF (no need for a separate `cdf_normal` function)

### `src/model.jl`
- Define `Model` struct with fields: `mu, beta, delta, alpha, b, Nl, s, prob, labor, Nk, maxK, curvK, Nk2, NT`
- Implement `create_model(; kwargs...)` — constructs Model with default parameters, calls `tauchen()` internally
- Implement `calculate_K0(m, r)` — capital demand given interest rate
- Implement `calculate_w0(m, r)` — wage given interest rate

### `src/grid.jl`
- Implement `generate_capital_grid(m, r=nothing, wage=nothing)` — creates state and control capital grids
  - When `r` and `wage` are `nothing`, use `phi = 0.0` (no-borrowing case for transition)
  - Otherwise compute borrowing limit from `min(b, wage*s[1]/r)`
- Implement `translate_capital_grid(m, grids)` — maps control grid points to nearby state grid points for interpolation
  - Returns `(kc1vec, kc2vec, prk1vec, prk2vec)`

### `src/vfi.jl`
- Implement `solve_VFI(m, r0, K0, wage, grids, capital_grid_translations; tau=0)`
  - Solves the household Bellman equation via value function iteration with grid search over `gridk2`
  - Budget constraint: `cons = s[lc]*wage + (1 + r0*(1-tau))*gridk[kc] - gridk2[kcc] + tau*r0*K0`
  - Converges when policy function indices stop changing
  - Returns `(kfun, kfunG, v)`: policy function (levels), policy function (grid indices on `gridk2`), value function

### `src/vfi_continuous.jl`
- Implement `solve_VFI_continuous(m, r0, K0, wage, grids, capital_grid_translations; tau=0)`
  - Same signature and return types as `solve_VFI` so it's a drop-in replacement
  - For each (kc, lc), uses **golden section search** to find optimal continuous k' in `[gridk2[1], kp_max]`, where `kp_max` is the largest k' that keeps consumption strictly positive
  - The continuation value is computed by linearly interpolating the next-period value function on `gridk` (the state grid)
  - Returns `(kfun, kfunG, v)` where `kfun` holds the continuous k' values and `kfunG` holds the nearest `gridk2` index (so `derive_stationary_distribution` works unchanged)
  - Convergence based on value-function distance: `maximum(abs.(v - v_old)) < 1e-6`
- Helper `golden_section_search(f, a, b; tol=1e-8)` — minimizes `f` on `[a, b]`
- Helper `interp_value(k_prime, gridk, v_row)` — linear interpolation of value on state grid
- Helper `find_nearest_grid_index(k_prime, gridk2)` — binary search for closest `gridk2` index

### `src/distribution.jl`
- Implement `derive_stationary_distribution(m, kfunG, capital_grid_translations)`
  - Iterates distribution forward using policy function until convergence
  - Returns the stationary distribution `mea0`
- Implement `calculate_capital_supply(mea0, kfun)` — computes aggregate capital supply as `sum(mea0 .* kfun)`

### `src/equilibrium.jl`
- Implement `solve_stationary_equilibrium(m; tau=0, adj=0.001, rate0=0.02, method=:grid_search)`
  - Iterates: compute K demand, solve HH problem, compute K supply
  - Stops when capital demand < capital supply
  - `method` can be `:grid_search` (calls `solve_VFI`) or `:continuous` (calls `solve_VFI_continuous`)
  - Returns named tuple `(; rate0, K1, w0, mea0, grid1, v)`
- Implement `derive_capital_supply_curve(m, r0_grid; tau=0, method=:grid_search)`
  - Computes capital supply for each interest rate in the grid
  - Returns named tuple `(; K0_grid, K1_grid)`

### `src/transition.jl`
- Implement `make_initial_guesses(m, K_SS0, K_SS1, tau)` — linear interpolation from old to new steady state
- Implement `compute_value_function_backwards!(vfun0, kfunGT, kfunT, m, tc, ...)` — one-period backward step (grid-search)
- Implement `compute_all_value_function_backwards(m, TT0, KT0, rT0, v_SS1, ...)` — full backward iteration (grid-search)
- Implement `compute_distribution_t(m, kfunGT, capital_grid_translations, mea_SS0)` — forward distribution iteration
- Implement `compute_capital(m, KT0, kfunT, meaT)` — aggregate capital path from distributions
- Implement `update_variables!(TT0, KT0, rT0, m, KT1, K_SS0, K_SS1, tau; adjK=0.04)` — update guesses with endpoint pinning
- Implement `derive_transition(m, K_SS0, K_SS1, v_SS1, mea_SS0, tau; method=:grid_search)` — main transition solver loop, dispatches on `method`

### `src/transition_continuous.jl`
- Implement `compute_value_function_backwards_continuous!(vfun0, kfunGT, kfunT, m, tc, ...)` — one-period backward step using golden section search (mirrors the grid-search version)
- Implement `compute_all_value_function_backwards_continuous(m, TT0, KT0, rT0, v_SS1, ...)` — full backward iteration with continuous control
- Reuses `golden_section_search`, `interp_value`, and `find_nearest_grid_index` from `vfi_continuous.jl`

### `Ch5/aiyagari.jl` (Main Script)
- Include all source files (including `vfi_continuous.jl` and `transition_continuous.jl`)
- Create model, solve stationary equilibrium, plot density and supply/demand curves
- Solve transition dynamics (tax change from 0 to 0.1), plot capital and interest rate paths
- All figures must be saved under `Ch5/images/` (e.g. `savefig("images/fig_density.pdf")`)
- To use continuous-control VFI, pass `method=:continuous` to `solve_stationary_equilibrium`, `derive_capital_supply_curve`, or `derive_transition`

## Key Differences from Reference Code

1. **Model struct**: Extended with `Nk, maxK, curvK, Nk2, NT` fields (reference code had a minimal struct)
2. **Grid generation**: Factored into separate functions (reference code had grids inline)
3. **Tax support**: VFI budget constraint includes `tau` and lump-sum transfer `tau*r0*K0`
4. **Capital grid max**: 30 instead of 20 (avoids distribution piling up at boundary)
5. **Transition endpoint pinning**: `update_variables!` enforces `KT0[1] = K_SS0` and `KT0[end] = K_SS1`
6. **Uses `Distributions.jl`** for normal CDF instead of a custom function
7. **Continuous-control VFI**: Provides a golden-section-search-based alternative to grid search

## Validation

After creating the files, verify that `Ch5/aiyagari.jl` can be run with `julia Ch5/aiyagari.jl` and produces:
- Stationary equilibrium with interest rate around 0.0285
- Transition dynamics converging within ~100 iterations
- Output plots saved as PDFs under `Ch5/images/`
- Both `method=:grid_search` and `method=:continuous` produce comparable equilibrium values
