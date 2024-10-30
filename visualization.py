from matplotlib import pyplot as plt
import pandas as pd

def plot_intervals(df, solution):
    fig, axes = plt.subplots(1, 2)
    
    # Plot each job as a horizontal bar spanning from release time to deadline
    for i, row in df.iterrows():
        start = row['Release Time']
        end = row['Deadline']
        axes[0].barh(i, end - start, left=start, color='skyblue', edgecolor='black')
        #print(i, row['Name'], start)
        
        axes[0].text((start + end) / 2, i, f'R:{start}, P:{row["Processing Time"]}, D:{end}, U:{row["Utility_1"]}', va='center', ha='center', fontsize=8, color='black')

    # Set labels and title
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Job Index')
    axes[0].set_title('Interval Scheduling: Release Time, Processing Time, and Deadline')

    # Set y-ticks as job indices
    axes[0].set_yticks(range(len(df)))

    axes[0].set_yticklabels([f'Job {df.iloc[i]["Name"]}' for i in range(len(df))])

    for i, interval in enumerate(solution):
        start = interval[1]
        end = interval[2]
        axes[1].barh(i, end - start, left=start, color='skyblue', edgecolor='black')
        
        axes[1].text((start + end) / 2, i, f'{interval[3]}', va='center', ha='center', fontsize=8, color='black')


    axes[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
    axes[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()



def plot_envy(envy):
    for agent in range(envy.shape[0]):
        plt.plot(envy[agent], label=f'Agent {agent}')

    plt.xlabel('Round')
    plt.ylabel('Envy')
    plt.title('Envy of Each Agent Over Successive Rounds')
    plt.legend()
    plt.show()