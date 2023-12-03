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

    # Define binary variables for each billboard and rank
    binary_vars = {}
    for b in billboards:
        for r in range(1, num_billboards + 1):
            var_name = f"x_{b}_{r}"
            binary_vars[var_name] = cpx.variables.add(names=[var_name], types=["B"])

    # Objective function
    for b in billboards:
        for r in range(1, num_billboards + 1):
            var_name = f"x_{b}_{r}"
            cpx.objective.set_linear([(var_name, sum(scores[b]) * r)])

    # Constraints: Each billboard has exactly one rank
    for b in billboards:
        cpx.linear_constraints.add(
            lin_expr=[[[f"x_{b}_{r}" for r in range(1, num_billboards + 1)], [1] * num_billboards]],
            senses=["E"],
            rhs=[1]
        )

    # Constraints: Each rank is assigned to exactly one billboard
    for r in range(1, num_billboards + 1):
        cpx.linear_constraints.add(
            lin_expr=[[[f"x_{b}_{r}" for b in billboards], [1] * num_billboards]],
            senses=["E"],
            rhs=[1]
        )

    # Solve the problem
    try:
        cpx.solve()
    except CplexError as exc:
        print(exc)

    # Output the solution
    solution = cpx.solution
    status = solution.get_status()

    if solution.is_primal_feasible():
        print(f"Solution status = {status}: {solution.status[status]}")
        print("Solution value =", solution.get_objective_value())
        # Determine the rank for each billboard
        ranks = {b: r for b in billboards for r in range(1, num_billboards + 1) if cpx.solution.get_values(f"x_{b}_{r}") == 1}
        # Output the ranks in the order of the billboards
        print("The ranks for billboard", ', '.join(billboards), "IS:", [ranks[b] for b in billboards])
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

def generate_random_scores(num_billboards):
    return {f'B{i+1}': [random.randint(1, 10) for _ in range(6)] for i in range(num_billboards)}


def analyze_ip_ranks(advertiser_preferences, base_prices):
    """
    Allocate billboards based on IP ranks, resolving conflicts with VCG auction.
    :param advertiser_preferences: DataFrame with advertisers as rows and billboards as columns showing rank preferences.
    :param base_prices: Dictionary with base prices for each billboard.
    :return: DataFrame showing the final allocation and prices.
    """
    allocation_records = []

    for billboard in advertiser_preferences.columns:
        # Get the base price for the current billboard
        current_base_price = base_prices[billboard]

        # Get the advertisers with the highest preference for this billboard
        top_preferences = advertiser_preferences[advertiser_preferences[billboard] == advertiser_preferences[billboard].min()]

        if len(top_preferences) == 1:  # No conflict
            advertiser = top_preferences.index[0]
            allocation_records.append({'Advertiser': advertiser, 'Billboard': billboard, 'Price': current_base_price})
        else:  # Conflicted billboard
            bids = {advertiser: np.random.uniform(1.5 * current_base_price, 2 * current_base_price) for advertiser in top_preferences.index}
            print(f"The bids variable: {bids}")
            winner, price = vcg_second_bid_auction(bids)
            allocation_records.append({'Advertiser': winner, 'Billboard': billboard, 'Price': price})

    return pd.DataFrame(allocation_records)

# Example usage
num_billboards = 4
num_advertisers = 4

# Initialize the DataFrame
advertiser_preferences = pd.DataFrame(index=[f'A{i+1}' for i in range(num_advertisers)],
                                      columns=[f'B{i+1}' for i in range(num_billboards)])

# Generate ranks for each advertiser
for i in range(num_advertisers):
    scores = generate_random_scores(num_billboards)
    ranks = find_advertiser_rank(num_billboards, scores)
    if ranks:
        for billboard, rank in ranks.items():
            advertiser_preferences.at[f'A{i+1}', billboard] = rank

# Set dynamic base prices for each billboard
base_prices = {
    'B1': 1000,
    'B2': 1200,
    'B3': 1100,
    'B4': 1050
}

# Analyze IP ranks and allocate billboards with dynamic base prices
final_allocation = analyze_ip_ranks(advertiser_preferences, base_prices)
print(final_allocation)
