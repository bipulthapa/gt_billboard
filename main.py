import random
import pandas as pd
import numpy as np

import cplex
from cplex.exceptions import CplexError

def find_advertiser_rank(num_billboards, scores):
    # Generate billboard names based on the number of billboards
    billboards = [f'B{i+1}' for i in range(num_billboards)]

    # Instantiate CPLEX problem
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)

    # Define binary variables and objective function only for billboards of interest
    for b in billboards:
        if scores[b] is not None:
            for r in range(1, num_billboards + 1):
                var_name = f"x_{b}_{r}"
                cpx.variables.add(names=[var_name], types=["B"])
                cpx.objective.set_linear([(var_name, sum(scores[b]) * r)])

    # Constraints for billboards of interest
    for b in billboards:
        if scores[b] is not None:
            rank_vars = [f"x_{b}_{r}" for r in range(1, num_billboards + 1)]
            cpx.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=rank_vars, val=[1.0] * len(rank_vars))],
                senses=["E"],
                rhs=[1]
            )

    # Ensure each rank is assigned to at most one billboard
    for r in range(1, num_billboards + 1):
        rank_vars = [f"x_{b}_{r}" for b in billboards if scores[b] is not None]
        cpx.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=rank_vars, val=[1.0] * len(rank_vars))],
            senses=["L"],
            rhs=[1]
        )

    # Solving the CPLEX problem
    try:
        cpx.solve()
    except CplexError as exc:
        print(exc)
        return None

    solution = cpx.solution
    status = solution.get_status()

    if solution.is_primal_feasible():
        ranks = {b: r for b in billboards for r in range(1, num_billboards + 1) if scores[b] is not None and cpx.solution.get_values(f"x_{b}_{r}") == 1}
        return ranks
    else:
        print("No solution available.")
        return None


def vcg_second_bid_auction(bids):
    """
    Conduct a VCG second-bid auction.
    :param bids: Dictionary with advertiser as key and their bid as value.
    :return: Tuple of winner and price to be paid.
    """
    sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
    winner, highest_bid = sorted_bids[0]
    second_highest_bid = sorted_bids[1][1] if len(sorted_bids) > 1 else highest_bid
    return winner, second_highest_bid

def generate_random_scores(num_billboards, advertiser_index, seed=None, ):
    """
    Generate the normalized score 0-10 for six factors.
    :param num_billboards: Number of available total billboards.
    :param seed: Seed value for random number generation.
    :return: Dictionary with advertiser as key and the factor score as value.
    """
    if seed is not None:
        random.seed(seed + advertiser_index)

    return {f'B{i+1}': [random.randint(1, 10) for _ in range(6)] if random.random() < 0.75 else None for i in range(num_billboards)}

def analyze_ip_ranks(advertiser_preferences, base_prices, seed=None):
    if seed is not None:
        np.random.seed(seed)

    allocation_records = []

    for billboard in advertiser_preferences.columns:
        current_base_price = base_prices[billboard]
        valid_preferences = advertiser_preferences[advertiser_preferences[billboard].notnull()]

        # Get advertisers with the highest preference
        if not valid_preferences.empty:
            top_preferences = valid_preferences[valid_preferences[billboard] == valid_preferences[billboard].min()]

            if len(top_preferences) == 1:
                advertiser = top_preferences.index[0]
                allocation_records.append({'Advertiser': advertiser, 'Billboard': billboard, 'Price': current_base_price})
            else:
                bids = {advertiser: np.random.uniform(1.5 * current_base_price, 2 * current_base_price) for advertiser in top_preferences.index}
                winner, price = vcg_second_bid_auction(bids)
                allocation_records.append({'Advertiser': winner, 'Billboard': billboard, 'Price': price})

    return pd.DataFrame(allocation_records)

num_billboards = 4
num_advertisers = 4
score_seed = 42
analyze_seed = 10

# Initialize the DataFrame
advertiser_preferences = pd.DataFrame(index=[f'A{i+1}' for i in range(num_advertisers)], columns=[f'B{i+1}' for i in range(num_billboards)])

# Set Dynamic base prices for each billboard by the owner
base_prices = {'B1': 1000, 'B2': 1200, 'B3': 1100, 'B4': 1050}

for i in range(num_advertisers):
    scores = generate_random_scores(num_billboards, i, score_seed)
    print(f"score: {scores}")
    ranks = find_advertiser_rank(num_billboards, scores)
    print(f"Advertiser Ranks: {ranks}")
    if ranks:
        for billboard, rank in ranks.items():
            advertiser_preferences.at[f'A{i+1}', billboard] = rank

# print(f"final advertiser_preferences: {advertiser_preferences}")
final_allocation = analyze_ip_ranks(advertiser_preferences, base_prices, analyze_seed)
print(final_allocation)

