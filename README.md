## Lok-MSDS-460-Assignment-2
MSDS 460 | Maddy Lok | 2 February 2025

# Network Models -- Project Management

## Overview
This project applies **Linear Programming (LP)** to determine the most optimal schedule for the development of a consumer-focused recommendation system for restaurants in Marlborough, Massachusetts. I utilized **Python and PuLP** to solve this problem in three different scenarios: best-case, worst-case, and the expected case scenarios. 

## Goal
- **Objective Function**: Minimize the total time of development for this project. 

## Methods
The project is implemented using **Python’s PuLP library**.

### **Part 2: Model Specification**
- **Objective Function**: Minimize the total time of development for this project. 
- **Critical Path Analysis**: No resource constraints were implemented. 
- **Decision Variables**: The start time of each task and the duration of each task.
- **Constraints**: The sequence dependency of each task and the total project time. 

### **Part 3: Programming**
- The optimal start times of each task was calculated based on estimated hours per task in best-case, worst-case, and expected-case scenarios. 
- The **critical path** was identified as: B → A → C → D1 → E → D2 → D3 → D4 → D6 → D5 → D7 → D8 → G → F → H

### **Part 4: Solution**
- The model was implemented in Python to solve for the minimized project development time for all three scenarios. 
- **Results**
  - Best-case scenario: 224 hours
  - Expected-case scenario: 300 hours
  - Worst-case scenario: 376 hours
- The critical path remained the same across all three scenarios, which implies that increases in the workforce affects tasks that are completed in parallel and does not have an impact on the sequential dependencies. 

### **Part 5: Overview**
**Expected number of hours for project completion:** 300 hours, or about 7.5 weeks. 
**Expected total labor costs:** $129,990

- If lower development times are necessary, increasing the workforce can decrease duration of project development, but will lead to increased costs. 

## Results Summary
- Development time of the recommendation system should take around 7.5 weeks as long as there are no unexpected delays.
- Increasing the workforce by 50% can result in the decrease in duration to around 6.2 weeks with increased total labor costs.
- This project is estimated to have total labor costs of about $129,990. 
