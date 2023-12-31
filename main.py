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
                all_bids[billboard].extend(bids.values())  # Add all bids to all_bids
                winner, price = vcg_second_bid_auction(bids)
                allocation_records.append({'Advertiser': winner, 'Billboard': billboard, 'Price': price})

    all_bids_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_bids.items()]))

    return pd.DataFrame(allocation_records), all_bids_df

def generate_dynamic_billboard_pricing(num_billboards, seed=None):
    """
    Generate dynamic pricing for billboards with realistic market values.
    :param num_billboards: Number of available billboards.
    :param seed: Seed value for random number generation.
    :return: Dictionary with billboard as key and dynamic price as value.
    """
    if seed is not None:
        random.seed(seed)

    # Realistic base pricing factors
    urban_base_price = 2000  # Base price for urban/high-traffic areas
    suburban_base_price = 1500  # Base price for suburban areas
    rural_base_price = 1000  # Base price for rural/low-traffic areas

    # Price variation range
    price_variation = 300  # Variation in price

    # Assigning prices based on hypothetical location types
    location_types = ['urban', 'suburban', 'rural']
    prices = {}
    for i in range(num_billboards):
        location = random.choice(location_types)
        base_price = urban_base_price if location == 'urban' else suburban_base_price if location == 'suburban' else rural_base_price
        prices[f'B{i+1}'] = base_price + random.randint(-price_variation, price_variation)

    return prices


num_billboards = 20
num_advertisers = 200
score_seed = 42
analyze_seed = 10

# Initialize the DataFrame
advertiser_preferences = pd.DataFrame(index=[f'A{i+1}' for i in range(num_advertisers)], columns=[f'B{i+1}' for i in range(num_billboards)])

all_bids = {billboard: [] for billboard in advertiser_preferences.columns}  # Initialize a dict to store all bids

# Set Dynamic base prices for each billboard by the owner
base_prices = generate_dynamic_billboard_pricing(num_billboards, analyze_seed)
# print(f"Base price: {base_prices}")

for i in range(num_advertisers):
    scores = generate_random_scores(num_billboards, i, score_seed)
    # print(f"score: {scores}")
    ranks = find_advertiser_rank(num_billboards, scores)
    # print(f"Advertiser Ranks: {ranks}")
    if ranks:
        for billboard, rank in ranks.items():
            advertiser_preferences.at[f'A{i+1}', billboard] = rank

# print(f"final advertiser_preferences: {advertiser_preferences}")
final_allocation, all_bids_df = analyze_ip_ranks(advertiser_preferences, base_prices, analyze_seed)
print(final_allocation)

