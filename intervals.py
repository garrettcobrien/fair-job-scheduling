import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bisect
import string

def generate_interval_scheduling_data(n, m, time_range=(1, 100), max_processing_time=10, utility_range=(1, 100)):
    # Generate random release times
    release_times = np.random.randint(time_range[0], time_range[1], size=n)
    
    # Generate random processing times
    processing_times = np.random.randint(1, max_processing_time, size=n)
    
    # Generate random deadlines, ensuring that the deadline is after the release time + processing time
    deadlines = release_times + processing_times + np.random.randint(1, max_processing_time, size=n)
    
    # Generate random utility vectors of size m with integer values
    utilities = np.random.randint(utility_range[0], utility_range[1], size=(n, m))

    #names
    letters = list(string.ascii_lowercase)

    # If there are more rows than letters, we'll cycle through the alphabet
    names = [letters[i % 26] for i in range(n)]
    
    # Create a DataFrame to store the data
    df = pd.DataFrame({
        'Name' : names,
        'Release Time': release_times,
        'Deadline': deadlines,
        'Processing Time': processing_times
    })
    
    # Add the utility vectors as columns to the DataFrame
    for i in range(m):
        df[f'Utility_{i+1}'] = utilities[:, i]
    
    return df

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

class Interval:
    def __init__(self, name, release, deadline, processing, utility) -> None:
        #release, deadline, processing, utility
        self.name = name
        self.release = release
        self.deadline = deadline
        self.processing = processing
        self.utility = utility

    def __lt__(self, other):
        if self.deadline == other.deadline:
            return self.deadline - self.processing < other.deadline - other.processing
        else:
            return self.deadline < other.deadline
        
    def __repr__(self):
        return f'(N:{self.name}, R:{self.release}, D:{self.deadline}, P:{self.processing}, U:{self.utility})'


class MWIS:
    def __init__(self, intervals: list[Interval]) -> None:
        self.intervals = intervals
        self.intervals.sort()

        self.opt = [-1] * len(intervals)

        self.schedule(len(self.intervals)-1, self.intervals[-1].deadline)

        print(self.opt)

        self.sol = []
        self.solution(len(self.intervals)-1, self.intervals[-1].deadline)


    def schedule(self, j, currentTime):
        if self.opt[j] != -1:
            return self.opt[j]
        
        #time it's actually scheduled
        scheduled_time = currentTime - self.intervals[j].processing

        #not compatible
        if scheduled_time < 0:
            self.opt[j] = 0
            return 0

        self.opt[j] = max(self.intervals[j].utility + self.schedule(self.compatible(j, scheduled_time), scheduled_time),
                        self.schedule(j-1, currentTime))

        return self.opt[j]


    def compatible(self, j, currentTime):
        while j >= 0:
            if self.intervals[j].release + self.intervals[j].processing <= currentTime:
                return j
            j -= 1

        return 0

    
    def solution(self, j, currentTime):
        scheduled_time = currentTime - self.intervals[j].processing
        if j != 0:
            if self.intervals[j].utility + self.opt[self.compatible(j, scheduled_time)] > self.opt[j-1]:
                #print(self.intervals[j])
                self.sol.append((j, scheduled_time, currentTime, self.intervals[j].name))
                
                self.solution(self.compatible(j, scheduled_time), scheduled_time)
            else:
                self.solution(j-1, currentTime)


if __name__ == "__main__":
    np.random.seed(42)
    # Parameters
    n = 15  # Number of intervals (jobs)
    m = 1  # Size of the utility vector
    time_range = (1, 25)
    max_processing_time = 10
    utility_range = (1, 10)  # Utility values will be integers between 1 and 100

    interval_data = generate_interval_scheduling_data(n, m, time_range, max_processing_time, utility_range)

    intervals = []
    for i, row in interval_data.iterrows():
        interval = Interval(row['Name'], row['Release Time'], row['Deadline'], row['Processing Time'], row['Utility_1'])
        intervals.append(interval)

    mwis = MWIS(intervals)


    interval_data['release_plus_processing'] = interval_data['Release Time'] + interval_data['Processing Time']

    plot_intervals(interval_data, mwis.sol)
