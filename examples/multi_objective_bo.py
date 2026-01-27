import time
import warnings

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def generate_initial_data(n=6, problem=None, NOISE_SE=None):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=n, q=1).squeeze(1)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj, problem=None, NOISE_SE=None):
    # define models for objective and constraint

    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i : i + 1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            SingleTaskGP(train_x, train_y, train_yvar)
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_qehvi_and_get_observation(
    model, train_x, train_obj, sampler, 
    problem=None, standard_bounds=None, BATCH_SIZE=None, 
    NUM_RESTARTS=None, RAW_SAMPLES=None, NOISE_SE=None
):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point,
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    problem = BraninCurrin(negate=True).to(**tkwargs)

    NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)

    BATCH_SIZE = 4
    NUM_RESTARTS = 10
    RAW_SAMPLES = 512

    standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
    standard_bounds[1] = 1

    N_BATCH = 2
    MC_SAMPLES = 4

    verbose = True

    hvs_qehvi = []

    # call helper functions to generate initial training data and initialize model
    train_x, train_obj, train_obj_true = generate_initial_data(
        n=2 * (problem.dim + 1), problem=problem, NOISE_SE=NOISE_SE
    )
    mll, model = initialize_model(train_x, train_obj, problem=problem, NOISE_SE=NOISE_SE)

    # compute hypervolume
    bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true)
    volume = bd.compute_hypervolume().item()

    hvs_qehvi.append(volume)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, N_BATCH + 1):

        t0 = time.monotonic()

        # fit the model
        fit_gpytorch_mll(mll)

        # define the qEHVI acquisition module using a QMC sampler
        qehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))

        # optimize acquisition function and get new observations
        new_x, new_obj, new_obj_true = optimize_qehvi_and_get_observation(
            model, train_x, train_obj, qehvi_sampler,
            problem=problem, standard_bounds=standard_bounds,
            BATCH_SIZE=BATCH_SIZE, NUM_RESTARTS=NUM_RESTARTS,
            RAW_SAMPLES=RAW_SAMPLES, NOISE_SE=NOISE_SE
        )

        # update training points
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        train_obj_true = torch.cat([train_obj_true, new_obj_true])

        # compute hypervolume
        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true)
        volume = bd.compute_hypervolume().item()
        hvs_qehvi.append(volume)

        # reinitialize the model so it is ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll, model = initialize_model(train_x, train_obj, problem=problem, NOISE_SE=NOISE_SE)

        t1 = time.monotonic()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: Hypervolume (qEHVI) = "
                f"{hvs_qehvi[-1]:>4.2f}, "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")
