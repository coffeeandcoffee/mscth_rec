import pandas as pd
import numpy as np

# Read the CSV
df = pd.read_csv("../data-ipf-hpi/pre_and_post_survey.csv")

# Define columns to keep and their new names
col_mapping = {
    'ID': 'ID',
    'sex': 'Sex',
    'How old are you?': 'Age',
    'How much time per day do you spend watching short-form videos?': 'Daily SFV Use',
    'How many hours did you sleep last night?': 'Sleep (h)',
    'Before starting, how alert do you feel?': 'Alertness',
    'In the last 6 hours, have you consumed:': 'Consumed',
}

# Select and rename columns
df_subset = df[list(col_mapping.keys())].rename(columns=col_mapping)

# Clean up Sex (f -> F, m -> M)
df_subset['Sex'] = df_subset['Sex'].str.upper()

# Handle exclusions
excluded_ids = ['P16', 'P19', 'P29']
df_subset['Status'] = df_subset['ID'].apply(lambda x: 'Excluded' if x in excluded_ids else 'Valid')

# Clean up 'Consumed' column (shorten)
def shorten_consumed(x):
    if pd.isna(x) or x == 'None':
        return 'None'
    x = str(x)
    return x.replace('Caffeine, Nicotine', 'Caff+Nic').replace('Energy drinks', 'Energy Dr.')

df_subset['Consumed'] = df_subset['Consumed'].apply(shorten_consumed)

# Clean up SFV use
def shorten_sfv(x):
    x = str(x).replace(' minutes', 'm').replace(' hours', 'h').replace(' hour', 'h')
    x = x.replace('More than 3h', '>3h')
    return x
df_subset['Daily SFV Use'] = df_subset['Daily SFV Use'].apply(shorten_sfv)

# Generate LaTeX table
tex_lines = [
    "\\begin{table}[H]",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{",
    "\\begin{tabular}{llclllcl}",
    "\\toprule",
    "\\textbf{ID} & \\textbf{Sex} & \\textbf{Age} & \\textbf{Daily SFV Use} & \\textbf{Sleep (h)} & \\textbf{Alertness} & \\textbf{Consumed} & \\textbf{Status} \\\\",
    "\\midrule"
]

for _, row in df_subset.iterrows():
    # If excluded, maybe grey out or just list it
    row_str = f"{row['ID']} & {row['Sex']} & {row['Age']} & {row['Daily SFV Use']} & {row['Sleep (h)']} & {row['Alertness']} & {row['Consumed']} & {row['Status']} \\\\"
    tex_lines.append(row_str)

tex_lines.extend([
    "\\bottomrule",
    "\\end{tabular}",
    "}",
    "\\caption{Participant demographics and pre-session survey responses. SFV = Short-Form Video. P16, P19, and P29 were excluded from the final analysis.}",
    "\\label{tab:participant_demographics}",
    "\\end{table}"
])

with open("participant_table.tex", "w") as f:
    f.write("\n".join(tex_lines))

print("LaTeX table generated successfully.")
