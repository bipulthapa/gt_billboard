from main import advertiser_preferences, final_allocation

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
plt.xlim(0, 50)

plt.xticks(range(0, 51, 5))


plt.tight_layout()

plt.savefig('./Results/final_auction_price_analysis.png', format='png', bbox_inches='tight')


