# Assignment 2: Network Models – Project Management
Maddy Lok | MSDS 460 | 2 February 2025

### Part 1: Problem Setup

import pandas as pd

project_plan_path = "/Users/spud/Documents/MSDS 460/Lok-MSDS-460-Assignment-2/Lok-MSDS-460-Assignment-2/project-plan-v003.xlsx"
xls = pd.ExcelFile(project_plan_path)

xls.sheet_names

df_tasks = pd.read_excel(xls, sheet_name="Sheet1")

df_tasks.head(), df_tasks.columns

# replacing NaN values with 0
df_tasks[['bestCaseHours', 'expectedHours', 'worstCaseHours']] = df_tasks[['bestCaseHours', 'expectedHours', 'worstCaseHours']].fillna(0)
df_tasks[['bestCaseHours', 'expectedHours', 'worstCaseHours']].describe()

# defining estimated industry standard hours for each task 
time_estimates = {
    "A": (8, 12, 16),    # Describe product
    "B": (12, 16, 24),   # Develop marketing strategy
    "C": (16, 24, 32),   # Design brochure
    "D": (160, 200, 240),# Develop product prototype
    "D1": (24, 32, 40),  # Requirements analysis
    "D2": (40, 50, 60),  # Software design
    "D3": (40, 50, 60),  # System design
    "D4": (40, 50, 60),  # Coding
    "D5": (40, 50, 60),  # Write documentation
    "D6": (24, 32, 40),  # Unit testing
    "D7": (24, 32, 40),  # System testing
    "D8": (16, 24, 32),  # Package deliverables
    "E": (16, 24, 32),   # Survey potential market
    "F": (24, 32, 40),   # Develop pricing plan
    "G": (8, 12, 16),    # Develop implementation plan 
    "H": (16, 24, 32)    # Wrtie client proposal
}

for task_id, (best, expected, worst) in time_estimates.items():
    df_tasks.loc[df_tasks["taskID"] == task_id, ["bestCaseHours", "expectedHours", "worstCaseHours"]] = best, expected, worst

df_tasks.head()

# was having issues finding critical path so this is a debugging code i got from chatGPT
print("\nDEBUG: Industry-Standard Estimated Hours for Each Task")
for task, (best, expected, worst) in time_estimates.items():
    print(f"{task}: Best={best}, Expected={expected}, Worst={worst}")

# Task Dependency Graph
import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes (tasks)
for _, row in df_tasks.iterrows():
    G.add_node(row['taskID'], label=row['task'])

# Add edges (dependencies)
for _, row in df_tasks.iterrows():
    if pd.notna(row['predecessorTaskIDs']):  # If there are dependencies
        predecessors = str(row['predecessorTaskIDs']).split(',')
        for pred in predecessors:
            pred = pred.strip()
            if pred in G.nodes:
                G.add_edge(pred, row['taskID'])
# Plot the dependency graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # Position nodes for visualization
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="orange", edge_color="gray", font_size=10, font_weight="bold", arrows=True)
plt.title("Task Dependency Graph")
plt.show()

independent_tasks = df_tasks[df_tasks["predecessorTaskIDs"].isna()]["taskID"].tolist()

single_dependency_tasks = df_tasks[df_tasks["predecessorTaskIDs"].str.count(",").fillna(0) == 0]["taskID"].tolist()
parallel_tasks = {
    "Independent Tasks (Start Immediately)": independent_tasks,
    "Single Dependency Tasks (Can Start Soon)": single_dependency_tasks,
}

parallel_tasks

## Part 2: Model Specification & Part 3: Programming

from pulp import LpMinimize, LpProblem, LpVariable

# LP Problem
lp_model = LpProblem("Project_Scheduling_Optimization", LpMinimize)

# decision variables
task_vars = {task: LpVariable(f"T_{task}", lowBound=0) for task in df_tasks["taskID"]}

# constraint: total project time
T_end = LpVariable("T_end", lowBound=0)

# objective function: minimize total duration of development time
lp_model += T_end

# constraint: dependency constraint
for _, row in df_tasks.iterrows():
    task = row["taskID"]
    duration = row["expectedHours"]

    if pd.notna(row["predecessorTaskIDs"]): 
        predecessors = str(row["predecessorTaskIDs"]).split(",")
        for pred in predecessors:
            pred = pred.strip()
            if pred in task_vars: 
                lp_model += task_vars[pred] + duration <= task_vars[task]

for task, var in task_vars.items():
    duration = df_tasks.loc[df_tasks["taskID"] == task, "expectedHours"].values[0]
    lp_model += var + duration <= T_end

# solving problem
from pulp import PULP_CBC_CMD
lp_model.solve(PULP_CBC_CMD(msg=False))

print("Project Scheduling Optimization Results:")
print(f"Minimum Project Completion Time: {T_end.varValue} hours\n")

print("Task Start Times:")
for task, var in task_vars.items():
    print(f"{task}: Start at {var.varValue} hours")
    
    # task start and end times
task_end_times = {task: var.varValue + df_tasks.loc[df_tasks["taskID"] == task, "expectedHours"].values[0] for task, var in task_vars.items()}
task_start_times = {task: task_vars[task].varValue for task in task_vars}

# another debugging code i got from chatgpt to try to fix my critical path code
h_slack = task_vars["H"].varValue - max(task_end_times["F"], task_end_times["G"])
print(f"\nDEBUG: Task H Slack = {h_slack} hours")

lp_model += task_vars["H"] == task_end_times["F"]

critical_path = set()

def trace_full_critical_path(task):
    """Recursively add tasks to the critical path by following dependencies back to the beginning."""
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

critical_path = sorted(set(critical_path), key=lambda x: task_vars[x].varValue)

print("\nCritical Path Tasks:")
print(" → ".join(critical_path))


lp_model

## Part 4: Solution

# Gantt chart overall
plt.figure(figsize=(12, 6))

for i, task in enumerate(task_vars.keys()):
    start = task_start_times[task]
    duration = df_tasks.loc[df_tasks["taskID"] == task, "expectedHours"].values[0]

    color = "red" if task in critical_path else "navy"
    
    plt.barh(i, duration, left=start, color=color)

plt.xlabel("Hours")
plt.ylabel("Tasks")
plt.title("Project Schedule Gantt Chart")
plt.yticks(range(len(task_vars)), list(task_vars.keys()))
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

# solving lp problem for all three scenarios
def solve_and_visualize_scenario(scenario_name, time_column):
    """Solves the LP model for the given scenario (best-case, expected-case, worst-case) and generates a Gantt chart."""
    
    lp_model = LpProblem(f"Project_Scheduling_{scenario_name}", LpMinimize)
    task_vars = {task: LpVariable(f"T_{task}", lowBound=0) for task in df_tasks["taskID"]}

    T_end = LpVariable("T_end", lowBound=0)

    lp_model += T_end

    for _, row in df_tasks.iterrows():
        task = row["taskID"]
        duration = row[time_column] 

        if pd.notna(row["predecessorTaskIDs"]): 
            predecessors = str(row["predecessorTaskIDs"]).split(",")
            for pred in predecessors:
                pred = pred.strip()
                if pred in task_vars:
                    lp_model += task_vars[pred] + duration <= task_vars[task]

    for task, var in task_vars.items():
        duration = df_tasks.loc[df_tasks["taskID"] == task, time_column].values[0]
        lp_model += var + duration <= T_end

    lp_model.solve(PULP_CBC_CMD(msg=False))

    task_start_times = {task: task_vars[task].varValue for task in task_vars}
    task_end_times = {task: task_start_times[task] + df_tasks.loc[df_tasks["taskID"] == task, time_column].values[0] for task in task_vars}

    critical_path = set()

    def trace_critical_path(task):
        """Recursively add tasks to the critical path by following dependencies."""
        if task in critical_path:
            return
        critical_path.add(task)

        predecessors = df_tasks.loc[df_tasks["taskID"] == task, "predecessorTaskIDs"].values[0]
        if pd.notna(predecessors):
            for pred in str(predecessors).split(","):
                pred = pred.strip()
                if pred in task_end_times:
                    trace_critical_path(pred)

    for task, end_time in task_end_times.items():
        if abs(end_time - T_end.varValue) <= 2:
            trace_critical_path(task)

    critical_path = sorted(set(critical_path), key=lambda x: task_vars[x].varValue)

    print(f"\n{scenario_name} Scenario")
    print(f"Project Completion Time: {T_end.varValue} hours")
    print("Critical Path:", " → ".join(critical_path))

    plt.figure(figsize=(12, 6))
    for i, task in enumerate(task_vars.keys()):
        start = task_start_times[task]
        duration = df_tasks.loc[df_tasks["taskID"] == task, time_column].values[0]
        color = "brown" if task in critical_path else "lightblue"
        plt.barh(i, duration, left=start, color=color)

    plt.xlabel("Hours")
    plt.ylabel("Tasks")
    plt.title(f"{scenario_name} Scenario")
    plt.yticks(range(len(task_vars)), list(task_vars.keys()))
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()

solve_and_visualize_scenario("Best-Case", "bestCaseHours")
solve_and_visualize_scenario("Expected-Case", "expectedHours")
solve_and_visualize_scenario("Worst-Case", "worstCaseHours")


# estimated hourly rates for contractors in each role per the averages of the state of massachusetts
hourly_rates = {
    "Project Manager": 60, 
    "Frontend Dev": 55, 
    "Backend Dev": 65, 
    "Data Scientist": 70, 
    "Data Engineer": 50
}

# estimated number of workers in each role for each task
worker_requirements = {
    "A": {"Project Manager": 1}, 
    "B": {"Project Manager": 1}, 
    "C": {"Frontend Dev": 1}, 
    "D1": {"Project Manager": 1, "Backend Dev": 2, "Data Scientist": 2, "Data Engineer": 2}, 
    "D2": {"Data Engineer": 1, "Backend Dev": 2}, 
    "D3": {"Backend Dev": 3, "Data Scientist": 2, "Data Engineer": 1}, 
    "D4": {"Backend Dev": 3, "Frontend Dev": 2, "Data Scientist": 2, "Data Engineer": 2}, 
    "D5": {"Frontend Dev": 3}, 
    "D6": {"Backend Dev": 1, "Data Scientist": 1, "Data Engineer": 2}, 
    "D7": {"Backend Dev": 3, "Frontend Dev": 2, "Data Scientist": 1, "Data Engineer": 1}, 
    "D8": {"Backend Dev": 2, "Frontend Dev": 1},  
    "E": {"Project Manager": 2, "Data Scientist": 2}, 
    "F": {"Project Manager": 3, "Data Scientist": 2}, 
    "G": {"Project Manager": 2, "Data Scientist": 1, "Data Engineer": 1}, 
    "H": {"Project Manager": 2, "Data Scientist": 1}
}

lp_model.solve(PULP_CBC_CMD(msg=False))

def calculate_project_cost(time_column, scenario_name):
    """Calculate total project cost based on best, expected, or worst-case task durations."""
    total_cost = 0
    task_costs = {}

    for task, workers in worker_requirements.items():
        duration = df_tasks.loc[df_tasks["taskID"] == task, time_column].values[0]
        task_cost = 0
        
        for role, count in workers.items():
            task_cost += count * hourly_rates[role] * duration

        task_costs[task] = task_cost
        total_cost += task_cost

    print(f"\n{scenario_name} Scenario - Total Estimated Project Cost: ${total_cost:,.2f}")
    
    return task_costs

best_case_costs = calculate_project_cost("bestCaseHours", "Best-Case")
expected_case_costs = calculate_project_cost("expectedHours", "Expected-Case")
worst_case_costs = calculate_project_cost("worstCaseHours", "Worst-Case")

def print_task_cost_breakdown(task_costs, scenario_name):
    """Prints the cost breakdown per task for a given scenario."""
    print(f"\n{scenario_name} Scenario - Cost Breakdown Per Task:")
    for task, cost in task_costs.items():
        print(f"{task}: ${cost:,.2f}")

print_task_cost_breakdown(best_case_costs, "Best-Case")
print_task_cost_breakdown(expected_case_costs, "Expected-Case")
print_task_cost_breakdown(worst_case_costs, "Worst-Case")

