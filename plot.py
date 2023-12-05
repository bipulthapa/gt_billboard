from main import advertiser_preferences, final_allocation, all_bids_df, base_prices, num_advertisers

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np

def billboard_sort_key(name):
    """
    Function to extract numerical values from the billboard names and convert them to integers
    """
    return int(name.strip('B'))

sorted_billboard_names = sorted(advertiser_preferences.columns, key=billboard_sort_key)

advertiser_preferences = advertiser_preferences.reindex(columns=sorted_billboard_names)

advertiser_preferences = advertiser_preferences.apply(pd.to_numeric, errors='coerce')

sorted_columns = sorted(advertiser_preferences.columns, key=lambda x: int(x[1:]))
advertiser_preferences_sorted = advertiser_preferences[sorted_columns]

# Competition Heatmap
competition = advertiser_preferences_sorted.count()
plt.figure(figsize=(12, 8))
sns.heatmap(competition.to_frame().T, annot=True, fmt='d', cmap='Reds', cbar_kws={'label': 'Number of Bids'})
plt.title('Billboard Competition Heatmap')
plt.xlabel('Billboard')
plt.ylabel('Number of Bids')
plt.savefig('./Results/billboard_competition_heatmap.png', bbox_inches='tight')

desirability = advertiser_preferences_sorted.mean()

# Desirability Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(desirability.to_frame().T, annot=True, cmap='Blues_r', cbar_kws={'label': 'Average Rank (Lower is Better)'})
plt.title('Billboard Desirability Heatmap')
plt.xlabel('Billboard')
plt.ylabel('Average Rank')
plt.savefig('./Results/billboard_desirability_heatmap.png', bbox_inches='tight')


sns.set_style("whitegrid")

final_allocation['Advertiser Index'] = final_allocation['Advertiser'].str.extract('(\d+)').astype(int)

final_allocation.sort_values('Advertiser Index', inplace=True)

plt.figure(figsize=(14, 8), dpi=300)
plt.scatter(final_allocation['Advertiser Index'], final_allocation['Price'],
            alpha=0.7, edgecolors='w', s=300, cmap='viridis')

plt.title('Bid Analysis: Advertiser Index vs. Final Auction Prices', fontsize=18)
plt.xlabel('Advertiser', fontsize=14)
plt.ylabel('Final Auction Prices', fontsize=14)

# Annotate each point with the corresponding billboard identifier
for idx, row in final_allocation.iterrows():
    plt.annotate(row['Billboard'], (row['Advertiser Index'], row['Price']),
                 textcoords="offset points",  # Offset the annotations slightly
                 xytext=(0,10),  # Distance from text to points (x,y)
                 ha='center',  # Horizontal alignment
                 fontsize=9)  # Font size for annotations

ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.tick_params(axis='both', which='major', labelsize=12)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
plt.xlim(0, num_advertisers + 1)

plt.xticks(range(0, num_advertisers + 1, 5))


plt.tight_layout()

plt.savefig('./Results/final_auction_price_analysis.png', format='png', bbox_inches='tight')

winning_bids = final_allocation.set_index('Billboard')['Price']

all_bids_df = all_bids_df.applymap(lambda x: np.nan if x is None else x)

min_bids = all_bids_df.min()
max_bids = all_bids_df.max()
max_bid_value = np.nanmax(all_bids_df.max())
horizontal_offset = 0.2  # horizontal offset
vertical_offset = max_bid_value * 0.05  # 5% of the max bid value for vertical spacing

plt.figure(figsize=(14, 8))

# Custom legend handles
range_line = mlines.Line2D([], [], color='grey', marker='o', linestyle='-', label='Bid Range')
winning_bid_marker = plt.Line2D([], [], color='red', marker='o', linestyle='None', label='Winning Bid')
default_price_marker = plt.Line2D([], [], color='blue', marker='s', linestyle='None', label='Default Base Price (No Competition)')

# Loop through each billboard and plot the range, the winning bid, and handle NaN cases
for idx, billboard in enumerate(all_bids_df.columns):
    if all_bids_df[billboard].isna().all():
        plt.scatter([idx], [base_prices[billboard]], color='blue', marker='s')
        plt.text(idx, base_prices[billboard], f' ${base_prices[billboard]:,.2f}',
                 ha='center', va='bottom', color='blue', fontsize=8)
    else:
        min_bid = min_bids[billboard]
        max_bid = max_bids[billboard]
        plt.plot([idx, idx], [min_bid, max_bid], marker='o', color='grey', zorder=1)
        plt.scatter([idx], [winning_bids[billboard]], color='red', zorder=2)

        plt.text(idx + horizontal_offset, winning_bids[billboard], f' ${winning_bids[billboard]:,.2f}',
             ha='left', va='center', color='blue', fontsize=8)

        # plt.text(idx, min_bids[billboard] - vertical_offset, f' ${min_bids[billboard]:,.2f}',
        #         ha='center', va='top', color='green', fontsize=8)

        # plt.text(idx, max_bids[billboard] + vertical_offset, f' ${max_bids[billboard]:,.2f}',
        #         ha='center', va='bottom', color='orange', fontsize=8)

# Labeling the plot
plt.title('Bid Range and Winning Bid for Each Billboard', fontsize=16)
plt.xlabel('Billboard', fontsize=14)
plt.ylabel('Bid Amount ($)', fontsize=14)
plt.xticks(range(len(all_bids_df.columns)), all_bids_df.columns)  # Set the x-ticks to be the billboard names
plt.grid(True)

plt.legend(handles=[range_line, winning_bid_marker, default_price_marker], loc='upper right')
plt.savefig('./Results/bid_range_with_winning_bid.png', format='png', bbox_inches='tight')