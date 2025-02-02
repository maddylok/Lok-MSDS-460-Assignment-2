import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pulp import LpMinimize, LpProblem, LpVariable, PULP_CBC_CMD

# Load project data
file_path = "project-plan-v003.xlsx"  # Replace with actual path
df_tasks = pd.read_excel(file_path, sheet_name="Sheet1")

# Fill missing values and ensure proper formatting
df_tasks[['bestCaseHours', 'expectedHours', 'worstCaseHours']] = df_tasks[['bestCaseHours', 'expectedHours', 'worstCaseHours']].fillna(0)

# Assign industry-standard estimated hours for each task
time_estimates = {
    "A": (8, 12, 16), "B": (12, 16, 24), "C": (16, 24, 32), "D": (160, 200, 240),
    "D1": (24, 32, 40), "D2": (40, 50, 60), "D3": (40, 50, 60), "D4": (40, 50, 60),
    "D5": (40, 50, 60), "D6": (24, 32, 40), "D7": (24, 32, 40), "D8": (16, 24, 32),
    "E": (16, 24, 32), "F": (24, 32, 40), "G": (8, 12, 16), "H": (16, 24, 32)
}

# Update the dataframe with the new estimates
for task_id, (best, expected, worst) in time_estimates.items():
    df_tasks.loc[df_tasks["taskID"] == task_id, ["bestCaseHours", "expectedHours", "worstCaseHours"]] = best, expected, worst

# Create the LP problem
lp_model = LpProblem("Project_Scheduling_Optimization", LpMinimize)

# Define decision variables (start times for each task)
task_vars = {task: LpVariable(f"T_{task}", lowBound=0) for task in df_tasks["taskID"]}

# Define the project completion time variable
T_end = LpVariable("T_end", lowBound=0)

# Set the objective function: Minimize project completion time
lp_model += T_end

# Add constraints for dependencies
for _, row in df_tasks.iterrows():
    task = row["taskID"]
    duration = row["expectedHours"]

    if pd.notna(row["predecessorTaskIDs"]):  # Check if dependencies exist
        predecessors = str(row["predecessorTaskIDs"]).split(",")
        for pred in predecessors:
            pred = pred.strip()
            if pred in task_vars:  # Ensure valid task ID
                lp_model += task_vars[pred] + duration <= task_vars[task]

# Ensure T_end is greater than all task completion times
for task, var in task_vars.items():
    duration = df_tasks.loc[df_tasks["taskID"] == task, "expectedHours"].values[0]
    lp_model += var + duration <= T_end

# Solve the LP problem
lp_model.solve(PULP_CBC_CMD(msg=False))

# Extract task start times
task_start_times = {task: task_vars[task].varValue for task in task_vars}
task_end_times = {task: task_start_times[task] + df_tasks.loc[df_tasks["taskID"] == task, "expectedHours"].values[0] for task in task_vars}

# Identify the critical path
critical_path = set()

def trace_full_critical_path(task):
    """Recursively add tasks to the critical path by following dependencies back to the start."""
    if task in critical_path:
        return

    critical_path.add(task)

    predecessors = df_tasks.loc[df_tasks["taskID"] == task, "predecessorTaskIDs"].values[0]
    if pd.notna(predecessors):
        for pred in str(predecessors).split(","):
            pred = pred.strip()
            if pred in task_end_times:
                trace_full_critical_path(pred)

for task, end_time in task_end_times.items():
    if abs(end_time - T_end.varValue) <= 2:
        trace_full_critical_path(task)

# Sort tasks by start time
critical_path = sorted(set(critical_path), key=lambda x: task_vars[x].varValue)

print("\nFINAL Critical Path Tasks:")
print(" → ".join(critical_path))

# Generate Gantt Chart
plt.figure(figsize=(12, 6))

for i, task in enumerate(task_vars.keys()):
    start = task_start_times[task]
    duration = df_tasks.loc[df_tasks["taskID"] == task, "expectedHours"].values[0]
    
    plt.barh(i, duration, left=start, color="red" if task in critical_path else "blue")

plt.xlabel("Hours")
plt.ylabel("Tasks")
plt.title("Project Schedule Gantt Chart")
plt.yticks(range(len(task_vars)), list(task_vars.keys()))
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()
