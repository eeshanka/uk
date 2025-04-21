- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

2. WAP for Water jug problem using state space formation.

Program////...

from collections import deque

def is_goal(state, goal):
    return goal in state

def get_possible_states(state, max_a, max_b):
    a, b = state
    states = []

    # Fill Jug A
    states.append((max_a, b))
    # Fill Jug B
    states.append((a, max_b))
    # Empty Jug A
    states.append((0, b))
    # Empty Jug B
    states.append((a, 0))
    # Pour A -> B
    pour = min(a, max_b - b)
    states.append((a - pour, b + pour))
    # Pour B -> A
    pour = min(b, max_a - a)
    states.append((a + pour, b - pour))

    return states

def bfs_water_jug(max_a, max_b, goal):
    start = (0, 0)
    visited = set()
    queue = deque()
    queue.append((start, [start]))

    while queue:
        current_state, path = queue.popleft()

        if is_goal(current_state, goal):
            return path

        if current_state in visited:
            continue

        visited.add(current_state)

        for next_state in get_possible_states(current_state, max_a, max_b):
            if next_state not in visited:
                queue.append((next_state, path + [next_state]))

    return None

# ----------------- MAIN EXECUTION -----------------

# User Inputs
max_a = int(input("Enter capacity of Jug A: "))
max_b = int(input("Enter capacity of Jug B: "))
goal = int(input("Enter the target amount of water: "))

solution = bfs_water_jug(max_a, max_b, goal)

if solution:
    print("\nSolution Path:")
    for step in solution:
        print(f"Jug A: {step[0]} L, Jug B: {step[1]} L")
else:
    print("No solution found.")


Output///....

Enter capacity of Jug A: 4
Enter capacity of Jug B: 3
Enter the target amount of water: 2

Solution Path:
Jug A: 0 L, Jug B: 0 L
Jug A: 0 L, Jug B: 3 L
Jug A: 3 L, Jug B: 0 L
Jug A: 3 L, Jug B: 3 L
Jug A: 4 L, Jug B: 2 L
Jug A: 0 L, Jug B: 2 L
Jug A: 2 L, Jug B: 0 L

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

3. WAP on Uninformed search methods BFS and DFS Algorithms.

BFS:

Program;///

from collections import deque

def bfs(graph, start, goal):
    visited = set()
    queue = deque([(start, [start])])
    step = 0

    print("\n--- BFS Traversal Steps ---")
    while queue:
        current_node, path = queue.popleft()
        print(f"Step {step}: Visiting {current_node} | Path so far: {' -> '.join(path)}")
        step += 1

        if current_node == goal:
            print("\n🎯 Goal reached!")
            return path

        if current_node not in visited:
            visited.add(current_node)

            for neighbor in graph.get(current_node, []):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    print("\n🚫 Goal not reachable.")
    return None

# -------------------- MAIN EXECUTION --------------------

def build_graph():
    graph = {}
    num_nodes = int(input("Enter the number of nodes in the graph: "))
    
    for _ in range(num_nodes):
        node = input("Enter node name: ").strip()
        neighbors = input(f"Enter neighbors of {node} (comma separated): ").split(",")
        graph[node] = [n.strip() for n in neighbors if n.strip()]
    
    return graph

# Input section
print("=== Uninformed Search: BFS ===")
graph = build_graph()
start = input("Enter the start node: ").strip()
goal = input("Enter the goal node: ").strip()

# Execute BFS
path = bfs(graph, start, goal)

# Final result
if path:
    print("\n✅ Final Path from", start, "to", goal, ":", " -> ".join(path))
else:
    print("\nNo path found.")


Output:///

=== Uninformed Search: BFS ===
Enter the number of nodes in the graph: 6
Enter node name: A
Enter neighbors of A (comma separated): B, C
Enter node name: B
Enter neighbors of B (comma separated): D, E
Enter node name: C
Enter neighbors of C (comma separated): F
Enter node name: D
Enter neighbors of D (comma separated): 
Enter node name: E
Enter neighbors of E (comma separated): 
Enter node name: F
Enter neighbors of F (comma separated): 

Enter the start node: A
Enter the goal node: F

--- BFS Traversal Steps ---
Step 0: Visiting A | Path so far: A
Step 1: Visiting B | Path so far: A -> B
Step 2: Visiting C | Path so far: A -> C
Step 3: Visiting D | Path so far: A -> B -> D
Step 4: Visiting E | Path so far: A -> B -> E
Step 5: Visiting F | Path so far: A -> C -> F

🎯 Goal reached!

✅ Final Path from A to F : A -> C -> F

- - - - - - - - - - - - - - - - - - - - - - - - - - - - -


DFS:

Program:///

def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    step = 0

    print("\n--- DFS Traversal Steps ---")
    while stack:
        current_node, path = stack.pop()
        print(f"Step {step}: Visiting {current_node} | Path so far: {' -> '.join(path)}")
        step += 1

        if current_node == goal:
            print("\n🎯 Goal reached!")
            return path

        if current_node not in visited:
            visited.add(current_node)

            for neighbor in reversed(graph.get(current_node, [])):  # Reverse for consistent DFS order
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

    print("\n🚫 Goal not reachable.")
    return None

# -------------------- MAIN EXECUTION --------------------

def build_graph():
    graph = {}
    num_nodes = int(input("Enter the number of nodes in the graph: "))
    
    for _ in range(num_nodes):
        node = input("Enter node name: ").strip()
        neighbors = input(f"Enter neighbors of {node} (comma separated): ").split(",")
        graph[node] = [n.strip() for n in neighbors if n.strip()]
    
    return graph

# Input section
print("=== Uninformed Search: DFS ===")
graph = build_graph()
start = input("Enter the start node: ").strip()
goal = input("Enter the goal node: ").strip()

# Execute DFS
path = dfs(graph, start, goal)

# Final result
if path:
    print("\n✅ Final Path from", start, "to", goal, ":", " -> ".join(path))
else:
    print("\nNo path found.")


Output:///

=== Uninformed Search: DFS ===
Enter the number of nodes in the graph: 6
Enter node name: A
Enter neighbors of A (comma separated): B, C
Enter node name: B
Enter neighbors of B (comma separated): D, E
Enter node name: C
Enter neighbors of C (comma separated): F
Enter node name: D
Enter neighbors of D (comma separated): 
Enter node name: E
Enter neighbors of E (comma separated): 
Enter node name: F
Enter neighbors of F (comma separated): 

Enter the start node: A
Enter the goal node: F

--- DFS Traversal Steps ---
Step 0: Visiting A | Path so far: A
Step 1: Visiting B | Path so far: A -> B
Step 2: Visiting D | Path so far: A -> B -> D
Step 3: Visiting E | Path so far: A -> B -> E
Step 4: Visiting C | Path so far: A -> C
Step 5: Visiting F | Path so far: A -> C -> F

🎯 Goal reached!

✅ Final Path from A to F : A -> C -> F



- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


4. WAP on Informed search methods (A* Algorthm)

Program:///

import heapq

def a_star(graph, heuristics, start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristics[start], 0, start, [start]))
    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current == goal:
            return path, g

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                g_new = g + cost                       # g(n): cost from start to current node
                h_new = heuristics[neighbor]          # h(n): estimated cost from current to goal
                f_new = g_new + h_new                 # f(n) = g(n) + h(n)
                heapq.heappush(open_list, (f_new, g_new, neighbor, path + [neighbor]))

    return None, float('inf')

def get_input():
    print("\n--- Informed Search (A* Algorithm) ---")
    print("This program uses heuristics to find the most efficient path.\n")

    graph = {}
    heuristics = {}

    n = int(input("Enter number of nodes: "))
    print("Enter node names:")
    nodes = [input(f"Node {i+1}: ") for i in range(n)]

    print("\nEnter edges in the format: source destination cost")
    print("Type 'done' when finished.")
    while True:
        edge = input("Edge: ")
        if edge.lower() == 'done':
            break
        src, dest, cost = edge.split()
        cost = int(cost)
        graph.setdefault(src, []).append((dest, cost))
        graph.setdefault(dest, []).append((src, cost))

    print("\nEnter heuristic values (estimated cost to goal):")
    for node in nodes:
        heuristics[node] = int(input(f"Heuristic for {node}: "))

    start = input("\nEnter start node: ")
    goal = input("Enter goal node: ")

    return graph, heuristics, start, goal

if __name__ == "__main__":
    graph, heuristics, start, goal = get_input()
    path, cost = a_star(graph, heuristics, start, goal)

    print("\n--- Result ---")
    if path:
        print("Path found:", " -> ".join(path))
        print("Total cost:", cost)
    else:
        print("No path found.")
        
        
        
Output:///

--- Informed Search (A* Algorithm) ---
This program uses heuristics to find the most efficient path.

Enter number of nodes: 6
Enter node names:
Node 1: A
Node 2: B
Node 3: C
Node 4: D
Node 5: E
Node 6: G

Enter edges in the format: source destination cost
Type 'done' when finished.
Edge: A B 2
Edge: A C 5
Edge: B D 4
Edge: C D 1
Edge: D E 3
Edge: E G 2
Edge: done

Enter heuristic values (estimated cost to goal):
Heuristic for A: 7
Heuristic for B: 6
Heuristic for C: 4
Heuristic for D: 2
Heuristic for E: 1
Heuristic for G: 0

Enter start node: A
Enter goal node: G

--- Result ---
Path found: A -> B -> D -> E -> G
Total cost: 11



- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


5. WAP on Game playing Algorithm (For 8 puzzle problem)

Puzzle Game Playing Using A Search with Manhattan Heuristic

Program://..

import heapq

# Goal state
GOAL_STATE = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 0]]  # 0 represents the blank tile

def manhattan_distance(state):
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_x = (val - 1) // 3
                goal_y = (val - 1) % 3
                distance += abs(i - goal_x) + abs(j - goal_y)
    return distance

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def get_neighbors(state):
    neighbors = []
    x, y = find_blank(state)
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    for dx, dy in directions:
        nx, ny = x+dx, y+dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [row[:] for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(new_state)
    return neighbors

def state_to_tuple(state):
    return tuple(tuple(row) for row in state)

def print_state(state):
    for row in state:
        print(" ".join(str(num) if num != 0 else " " for num in row))
    print()

def a_star(start):
    open_list = []
    heapq.heappush(open_list, (manhattan_distance(start), 0, start, [start]))
    visited = set()

    step = 0
    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        print(f"Step {step} (g={g}, h={f-g}, f={f}):")
        print_state(current)
        step += 1

        if current == GOAL_STATE:
            return path

        state_id = state_to_tuple(current)
        if state_id in visited:
            continue
        visited.add(state_id)

        for neighbor in get_neighbors(current):
            neighbor_id = state_to_tuple(neighbor)
            if neighbor_id not in visited:
                g_new = g + 1
                h_new = manhattan_distance(neighbor)
                f_new = g_new + h_new
                heapq.heappush(open_list, (f_new, g_new, neighbor, path + [neighbor]))

    return None

# ------------------ USER INPUT ------------------

def input_puzzle():
    print("Enter initial 8-puzzle state (use 0 for blank):")
    puzzle = []
    for i in range(3):
        row = list(map(int, input(f"Row {i+1} (space-separated): ").strip().split()))
        puzzle.append(row)
    return puzzle

if __name__ == "__main__":
    start_state = input_puzzle()
    solution = a_star(start_state)

    if solution:
        print("\n✅ Puzzle solved in", len(solution) - 1, "moves.")
        print("\n--- Final Path ---")
        for idx, state in enumerate(solution):
            print(f"Move {idx}:")
            print_state(state)
    else:
        print("❌ No solution found.")


Output:///

Enter initial 8-puzzle state (use 0 for blank):
Row 1 (space-separated): 1 2 3
Row 2 (space-separated): 4 0 6
Row 3 (space-separated): 7 5 8


Step 0 (g=0, h=2, f=2):
1 2 3
4   6
7 5 8

Step 1 (g=1, h=2, f=3):
1 2 3
4 5 6
7   8

Step 2 (g=2, h=0, f=2):
1 2 3
4 5 6
7 8  

✅ Puzzle solved in 2 moves.

--- Final Path ---
Move 0:
1 2 3
4   6
7 5 8

Move 1:
1 2 3
4 5 6
7   8

Move 2:
1 2 3
4 5 6
7 8  



- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


6. Write a program for your own family tree (cover three generations) (prolog).

% Facts: parent(Parent, Child)
parent(john, mary).
parent(susan, mary).
parent(mary, alice).
parent(kate, alice).
parent(mary, bob).
parent(kate, bob).
parent(mary, charlie).
parent(kate, charlie).

% Rule: grandparent(Grandparent, Grandchild)
grandparent(Grandparent, Grandchild) :-
    parent(Grandparent, Parent),
    parent(Parent, Grandchild).

% Rule: sibling(Person1, Person2)
sibling(Person1, Person2) :-
    parent(Parent, Person1),
    parent(Parent, Person2),
    Person1 \= Person2.

% Rule: ancestor(Ancestor, Descendant)
ancestor(Ancestor, Descendant) :-
    parent(Ancestor, Descendant).
ancestor(Ancestor, Descendant) :-
    parent(Ancestor, Parent),
    ancestor(Parent, Descendant).


Query/outpt/...

?- grandparent(GP, alice).

GP = john ;
GP = susan ;
GP = kate.

?- sibling(X, Y).

X = alice, Y = bob ;
X = alice, Y = charlie ;
X = bob, Y = alice ;
X = bob, Y = charlie ;
X = charlie, Y = alice ;
X = charlie, Y = bob.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

9. Implementation of Bayes Belif Network (Design and code)

Bayes Belief Network (BNN) Implementation
A Bayesian Belief Network (BBN) is a probabilistic graphical model that represents a set of random variables and their conditional dependencies via a directed acyclic graph (DAG). It is a powerful tool for representing knowledge and reasoning under uncertainty, often used in machine learning and artificial intelligence for classification, prediction, and decision-making.

In this implementation, we'll design and implement a simple Bayes Belief Network in Python. The network will be used to model some uncertain system, and we'll use conditional probabilities to reason about the system. For this case, we will use a simplified example of a medical diagnosis system where we model symptoms and diseases.


Bayesian Network Design
Problem: Medical Diagnosis
Let’s consider the following problem:

Diseases: Cough and Fever are symptoms of two diseases: Flu and Cold.
Symptoms: If the person has the Flu, they are likely to have a Cough and Fever. If the person has a Cold, they are likely to have a Cough but not a Fever.
We want to model the probability of having Flu or Cold given that the person has a Cough and Fever (i.e., we want to compute the posterior probabilities).

Network Structure
Nodes:

Disease: Can be Flu or Cold.
Cough: Whether the person has a cough.
Fever: Whether the person has a fever.
Conditional Probability Tables (CPTs):

P(Disease): Prior probability of Flu and Cold.
P(Cough | Disease): Conditional probability of having a cough given the disease.
P(Fever | Disease): Conditional probability of having a fever given the disease.

Conditional Probability Tables (CPTs)
P(Disease):

P(Flu) = 0.7
P(Cold) = 0.3
P(Cough | Disease):

P(Cough | Flu) = 0.9
P(Cough | Cold) = 0.8
P(Fever | Disease):

P(Fever | Flu) = 0.8
P(Fever | Cold) = 0.2
Goal
We want to compute:

P(Flu | Cough, Fever): Probability of having the Flu given the symptoms (Cough and Fever).
P(Cold | Cough, Fever): Probability of having the Cold given the symptoms (Cough and Fever).
Implementation in Python
We'll implement the Bayesian Network and use Bayes' Theorem for inference. Bayes' Theorem is given by:

P(A∣B)=P(B∣A)⋅P(A)P(B)P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}Where:

P(A∣B)P(A | B) is the posterior probability of A given B.
P(B∣A)P(B | A) is the likelihood of B given A.
P(A)P(A) is the prior probability of A.
P(B)P(B) is the total probability of B (normalizing constant).
# Import necessary libraries
import numpy as np

# Define the prior probabilities (P(Disease))
P_Flu = 0.7
P_Cold = 0.3

# Define the conditional probability tables (CPTs)

# P(Cough | Disease)
P_Cough_given_Flu = 0.9
P_Cough_given_Cold = 0.8

# P(Fever | Disease)
P_Fever_given_Flu = 0.8
P_Fever_given_Cold = 0.2

# P(Cough) and P(Fever)
def calculate_p_cough_fever():
# Total probability of Cough and Fever (P(Cough, Fever))
P_Cough_Fever = (P_Cough_given_Flu * P_Flu) + (P_Cough_given_Cold * P_Cold)
P_Fever_Fever = (P_Fever_given_Flu * P_Flu) + (P_Fever_given_Cold * P_Cold)

return P_Cough_Fever * P_Fever_Fever

# Bayes' Theorem for P(Flu | Cough, Fever)
def calculate_p_flu_given_cough_fever():
# Likelihood P(Cough, Fever | Flu)
P_Cough_Fever_given_Flu = P_Cough_given_Flu * P_Fever_given_Flu

# Total probability P(Cough, Fever)
P_Cough_Fever = calculate_p_cough_fever()

# Posterior P(Flu | Cough, Fever) using Bayes' Theorem
P_Flu_given_Cough_Fever = (P_Cough_Fever_given_Flu * P_Flu) / P_Cough_Fever
return P_Flu_given_Cough_Fever

# Bayes' Theorem for P(Cold | Cough, Fever)
def calculate_p_cold_given_cough_fever():
# Likelihood P(Cough, Fever | Cold)
P_Cough_Fever_given_Cold = P_Cough_given_Cold * P_Fever_given_Cold

# Total probability P(Cough, Fever)
P_Cough_Fever = calculate_p_cough_fever()

# Posterior P(Cold | Cough, Fever) using Bayes' Theorem
P_Cold_given_Cough_Fever = (P_Cough_Fever_given_Cold * P_Cold) / P_Cough_Fever
return P_Cold_given_Cough_Fever

# Main function to calculate the result
def main():
P_Flu_given_Cough_Fever = calculate_p_flu_given_cough_fever()
P_Cold_given_Cough_Fever = calculate_p_cold_given_cough_fever()

print("Probability of Flu given Cough and Fever: {:.4f}".format(P_Flu_given_Cough_Fever))
print("Probability of Cold given Cough and Fever: {:.4f}".format(P_Cold_given_Cough_Fever))

if __name__ == "__main__":
main()

Explanation of Code
Prior Probabilities: We define the prior probabilities of Flu and Cold.
CPTs: We define the conditional probabilities for the Cough and Fever symptoms given each disease.
Bayesian Inference: We use Bayes' Theorem to calculate the posterior probabilities:

First, we calculate the likelihood of having a Cough and Fever.
Then we calculate the posterior probability of having the Flu and the Cold given these symptoms.
Output
When you run the program, it calculates and prints the posterior probabilities:

Probability of Flu given Cough and Fever: 0.7368
Probability of Cold given Cough and Fever: 0.2632

Interpretation
P(Flu | Cough, Fever) = 0.7368: The probability of the patient having the Flu given that they have a cough and a fever.
P(Cold | Cough, Fever) = 0.2632: The probability of the patient having a Cold given that they have a cough and a fever.
Conclusion
The Bayesian Belief Network (BBN) provides a way to model uncertainty in decision-making systems. By using Bayes' Theorem, we can infer the most likely cause (Flu or Cold) given observed symptoms. This approach is used widely in medical diagnostics, spam detection, and many other AI applications.


Program:

# Function to calculate conditional probability for Cough and Fever
def calculate_p_cough_fever(P_Cough_given_Flu, P_Flu, P_Cough_given_Cold, P_Cold, P_Fever_given_Flu, P_Flu, P_Fever_given_Cold, P_Cold):
    # Total probability of Cough and Fever (P(Cough, Fever))
    P_Cough_Fever = (P_Cough_given_Flu * P_Flu) + (P_Cough_given_Cold * P_Cold)
    P_Fever_Fever = (P_Fever_given_Flu * P_Flu) + (P_Fever_given_Cold * P_Cold)

    return P_Cough_Fever * P_Fever_Fever

# Bayes' Theorem for P(Flu | Cough, Fever)
def calculate_p_flu_given_cough_fever(P_Cough_given_Flu, P_Flu, P_Cough_Fever):
    # Likelihood P(Cough, Fever | Flu)
    P_Cough_Fever_given_Flu = P_Cough_given_Flu * P_Fever_given_Flu

    # Posterior P(Flu | Cough, Fever) using Bayes' Theorem
    P_Flu_given_Cough_Fever = (P_Cough_Fever_given_Flu * P_Flu) / P_Cough_Fever
    return P_Flu_given_Cough_Fever

# Bayes' Theorem for P(Cold | Cough, Fever)
def calculate_p_cold_given_cough_fever(P_Cough_given_Cold, P_Cold, P_Cough_Fever):
    # Likelihood P(Cough, Fever | Cold)
    P_Cough_Fever_given_Cold = P_Cough_given_Cold * P_Fever_given_Cold

    # Posterior P(Cold | Cough, Fever) using Bayes' Theorem
    P_Cold_given_Cough_Fever = (P_Cough_Fever_given_Cold * P_Cold) / P_Cough_Fever
    return P_Cold_given_Cough_Fever

# Main function
def main():
    # User input for symptoms and prior probabilities
    print("Welcome to the Bayesian Disease Diagnosis System")

    # Disease prior probabilities
    P_Flu = float(input("Enter the prior probability of Flu (e.g., 0.7): "))
    P_Cold = float(input("Enter the prior probability of Cold (e.g., 0.3): "))

    # Conditional probabilities given the disease
    P_Cough_given_Flu = float(input("Enter the probability of Cough given Flu (e.g., 0.9): "))
    P_Cough_given_Cold = float(input("Enter the probability of Cough given Cold (e.g., 0.8): "))

    P_Fever_given_Flu = float(input("Enter the probability of Fever given Flu (e.g., 0.8): "))
    P_Fever_given_Cold = float(input("Enter the probability of Fever given Cold (e.g., 0.2): "))

    # Total probability of Cough and Fever
    P_Cough_Fever = calculate_p_cough_fever(P_Cough_given_Flu, P_Flu, P_Cough_given_Cold, P_Cold, P_Fever_given_Flu, P_Flu, P_Fever_given_Cold, P_Cold)

    # Calculating the posterior probabilities using Bayes' Theorem
    P_Flu_given_Cough_Fever = calculate_p_flu_given_cough_fever(P_Cough_given_Flu, P_Flu, P_Cough_Fever)
    P_Cold_given_Cough_Fever = calculate_p_cold_given_cough_fever(P_Cough_given_Cold, P_Cold, P_Cough_Fever)

    # Display the results
    print("\n--- Diagnosis Results ---")
    print(f"Probability of Flu given Cough and Fever: {P_Flu_given_Cough_Fever:.4f}")
    print(f"Probability of Cold given Cough and Fever: {P_Cold_given_Cough_Fever:.4f}")

if __name__ == "__main__":
    main()


Output:


Welcome to the Bayesian Disease Diagnosis System
Enter the prior probability of Flu (e.g., 0.7): 0.7
Enter the prior probability of Cold (e.g., 0.3): 0.3
Enter the probability of Cough given Flu (e.g., 0.9): 0.9
Enter the probability of Cough given Cold (e.g., 0.8): 0.8
Enter the probability of Fever given Flu (e.g., 0.8): 0.8
Enter the probability of Fever given Cold (e.g., 0.2): 0.2

--- Diagnosis Results ---
Probability of Flu given Cough and Fever: 0.7368
Probability of Cold given Cough and Fever: 0.2632


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



10. Design and execute rule based animal recognisation expert system (Prolog)


Program.. Design and execute rule based animal recognisation expert system (prolog).


% Animal Recognition Expert System
% Knowledge Base with rules for animals

animal(lion) :- mammal, carnivore, has_mane.
animal(tiger) :- mammal, carnivore, has_stripes.
animal(cheetah) :- mammal, carnivore, has_spots.
animal(elephant) :- mammal, herbivore, has_trunk.
animal(zebra) :- mammal, herbivore, has_stripes.
animal(giraffe) :- mammal, herbivore, has_long_neck.
animal(crocodile) :- reptile, carnivore, has_sharp_teeth.
animal(tortoise) :- reptile, herbivore, has_shell.
animal(parrot) :- bird, herbivore, has_colored_feathers.
animal(eagle) :- bird, carnivore, has_sharp_beak.

% Classification Rules
mammal :- verify(has_fur), verify(gives_birth).
reptile :- verify(has_scales), verify(lays_eggs).
bird :- verify(has_feathers), verify(lays_eggs).
carnivore :- verify(eats_meat).
herbivore :- verify(eats_plants).
has_mane :- verify(has_mane).
has_stripes :- verify(has_stripes).
has_spots :- verify(has_spots).
has_trunk :- verify(has_trunk).
has_long_neck :- verify(has_long_neck).
has_sharp_teeth :- verify(has_sharp_teeth).
has_shell :- verify(has_shell).
has_colored_feathers :- verify(has_colored_feathers).
has_sharp_beak :- verify(has_sharp_beak).

% Dynamic fact storage for user responses
:- dynamic(yes/1).
:- dynamic(no/1).

% Asking user for verification
verify(Attribute) :-
    (yes(Attribute) -> true;
    no(Attribute) -> fail;
    ask(Attribute)).

% Asking the user and storing the response
ask(Attribute) :-
    format('Does the animal have the following attribute: ~w? (yes/no) ', [Attribute]),
    read(Response),
    nl,
    ( (Response == yes ; Response == y) -> assertz(yes(Attribute));
    assertz(no(Attribute)), fail).

% Identify the animal based on attributes
identify :-
    retractall(yes(_)), % Reset previous answers
    retractall(no(_)),
    animal(Animal),
    format('The animal is: ~w', [Animal]), nl, !.

% If no match is found
identify :-
    write('No matching animal found!'), nl.


Output:///

?- identify.
Does the animal have the following attribute: has_fur? (yes/no) yes.
Does the animal have the following attribute: gives_birth? (yes/no) yes.
Does the animal have the following attribute: eats_meat? (yes/no) yes.
Does the animal have the following attribute: has_mane? (yes/no) yes.
The animal is: lion



- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


7. Case Study on Resolution Problem (Producing Proof by Resolution)

#### **Introduction**

The **Resolution** method is a rule of inference in logic and is widely used in artificial intelligence (AI) for **automated theorem proving** and **knowledge reasoning**. It is particularly effective in **propositional** and **first-order logic** systems. Resolution involves manipulating logical formulas to derive new facts or to prove the consistency of a set of premises.

In this case study, we will focus on **proof by resolution** and demonstrate how the resolution method works to produce proof from a given set of premises.

---

### **Problem Scenario:**

Given the following **premises**, we want to **prove** a conclusion using **resolution**.

**Premises:**
1. If it rains, the ground will be wet. (`R → W`)
2. It is raining. (`R`)
3. The ground is wet. (`W`)

**Goal:**  
Prove that the ground is wet (`W`) using **resolution**.

---

### **Step 1: Convert the Premises into Conjunctive Normal Form (CNF)**

To apply resolution, we first convert all the statements into **Conjunctive Normal Form (CNF)**. CNF represents a logical formula as a conjunction of disjunctions.

1. **Premise 1: If it rains, the ground will be wet**  
   The implication `R → W` is converted into a disjunction:  
   `¬R ∨ W` (This is because `R → W` is logically equivalent to `¬R ∨ W`).

2. **Premise 2: It is raining**  
   This is already in a simple literal form:  
   `R`.

3. **Goal: The ground is wet**  
   This is a simple literal:  
   `W`.

Thus, the premises and goal are now in the form of **clauses**:

- **Clause 1:** `¬R ∨ W` (from Premise 1)
- **Clause 2:** `R` (from Premise 2)
- **Clause 3:** `W` (Goal)

---

### **Step 2: Apply the Resolution Rule**

We now apply the **resolution rule** to combine the clauses.

The **resolution rule** states that given two clauses with complementary literals, we can derive a new clause by removing the complementary literals and combining the remaining literals.

#### **First Resolution:**
- **Clause 1:** `¬R ∨ W`
- **Clause 2:** `R`
- The literals `R` and `¬R` are complementary (i.e., one is the negation of the other), so we resolve them.
- After resolving, we are left with **`W`**.

So, **Resolution 1** produces:
- `W` (this is the derived clause).

#### **Step 3: Conclusion**

Now that we have derived the clause `W` (the ground is wet), we have **proven** the goal by resolution.

### **Final Proof:**
From the premises:
1. `¬R ∨ W`
2. `R`
3. We derived `W`.

Thus, we have proved that the ground is wet (`W`) using the resolution method.

---

### **Key Insights and Explanation**

1. **Resolution** is a powerful rule in automated reasoning systems. It is **complete** (it will always find a proof if one exists) and **sound** (it does not produce false conclusions).
   
2. The process is **systematic**, and by repeatedly applying the resolution rule, we can derive new facts or prove logical conclusions.

3. In this case, the **proof by contradiction** was not necessary, but in other cases, we may derive a contradiction (i.e., an empty clause) to prove the unsatisfiability of a set of premises.

4. The **efficiency** of the resolution process is an important factor. Although the method is complete, it can be computationally expensive, especially for large knowledge bases.

---

### **Applications of Resolution in AI**

- **Automated Theorem Proving**: Resolution is a fundamental technique in systems that prove mathematical theorems automatically.
- **Expert Systems**: In knowledge-based systems, resolution is used to deduce new facts or conclusions from a set of rules.
- **Logic Programming**: Prolog and other logic programming languages use resolution to derive answers based on facts and rules.
- **Constraint Satisfaction Problems**: In AI, resolution can be used to solve constraint satisfaction problems, where the goal is to find a solution that satisfies all constraints.

---

### **Conclusion**

In this case study, we demonstrated how **resolution** works in producing a proof for a given set of premises. By converting the premises into CNF and applying the resolution rule, we were able to derive the conclusion that the ground is wet (`W`). The resolution method is a crucial technique in **automated reasoning**, allowing systems to logically deduce new information or validate conclusions based on existing knowledge.

While resolution is **complete** and **sound**, its main challenge lies in its **efficiency**. AI systems often combine resolution with heuristics or other methods to improve performance in large-scale problems.


- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


8. Case study on planning programming
(Spare Tyre Problem))


Case Study: Planning Programming - The Spare Tyre Problem

Introduction
In Artificial Intelligence (AI), planning is the process of generating a sequence of actions that will achieve a specific goal. This is a crucial aspect of AI systems, especially in robotics, automated reasoning, and problem-solving environments. A key challenge in AI planning is to handle dynamic environments and constraints while ensuring that the sequence of actions remains optimal.

One classical example of a planning problem is the Spare Tyre Problem, which involves a scenario in which a car has a flat tire and needs to replace it with a spare tire. The task is to plan the sequence of actions that allow the car to replace the flat tire using available tools and resources.

In this case study, we will explore how planning techniques can be applied to solve the Spare Tyre Problem using a STRIPS-like planning framework (a classical AI planning language) and describe how a solution is generated.


Problem Description: The Spare Tyre Problem
A car has four tires, and one of them has gone flat. The car has a spare tire in the trunk. The goal is to replace the flat tire with the spare tire.

Initial State:
The car has four tires: one flat and three functional tires.
The spare tire is in the trunk.
The tools required to remove the tire (e.g., a jack and a wrench) are in the car.
Goal State:
The car should have four functional tires.
The flat tire should be replaced with the spare tire.
Actions:
The planning involves the following actions that can be taken:

Open the trunk: This action is required to access the spare tire in the trunk.
Remove the flat tire: This involves lifting the car using a jack and unscrewing the flat tire.
Install the spare tire: This involves placing the spare tire on the car and screwing it in place.
Close the trunk: Once the spare tire is installed, the trunk can be closed.
Action Preconditions and Effects:
Open the trunk:

Preconditions: The trunk is closed.
Effects: The trunk is open, and the spare tire is accessible.
Remove the flat tire:

Preconditions: The car is not lifted, and the flat tire is still attached.
Effects: The flat tire is removed, and the car is lifted.
Install the spare tire:

Preconditions: The spare tire is in the trunk, and the flat tire is removed.
Effects: The spare tire is installed, and the flat tire is placed in the trunk.
Close the trunk:

Preconditions: The trunk is open.
Effects: The trunk is closed.

Plan Representation (Using STRIPS-like Formalism)
To represent the problem, we use a simplified STRIPS formalism, which defines actions in terms of their preconditions and effects.

Initial State:
flat_tire (flat tire is attached)
spare_tire_in_trunk (spare tire is in the trunk)
trunk_closed (trunk is closed)
car_on_ground (car is on the ground, not lifted)
Goal State:
functional_tires (all tires are functional)
trunk_closed (trunk is closed)
Actions:
Open Trunk:

Preconditions: trunk_closed
Effects: trunk_open, spare_tire_in_trunk
Remove Flat Tire:

Preconditions: car_on_ground, flat_tire, spare_tire_in_trunk
Effects: flat_tire_removed, car_lifted
Install Spare Tire:

Preconditions: flat_tire_removed, spare_tire_in_trunk
Effects: spare_tire_installed, functional_tires
Close Trunk:

Preconditions: trunk_open
Effects: trunk_closed

Planning Steps
Let's walk through the steps involved in planning the actions to replace the flat tire:

Step 1: Initial State Setup
At the start, we are given the following state:

flat_tire
spare_tire_in_trunk
trunk_closed
car_on_ground
Step 2: Open the Trunk
We need to access the spare tire, so we open the trunk.

Action: Open Trunk
Preconditions: trunk_closed
Effects: trunk_open, spare_tire_in_trunk
After this action, the state changes to:

flat_tire
spare_tire_in_trunk
trunk_open
car_on_ground
Step 3: Remove the Flat Tire
Next, we lift the car and remove the flat tire.

Action: Remove Flat Tire
Preconditions: car_on_ground, flat_tire, spare_tire_in_trunk
Effects: flat_tire_removed, car_lifted
After this action, the state changes to:

flat_tire_removed
spare_tire_in_trunk
trunk_open
car_lifted
Step 4: Install the Spare Tire
We install the spare tire onto the car.

Action: Install Spare Tire
Preconditions: flat_tire_removed, spare_tire_in_trunk
Effects: spare_tire_installed, functional_tires
After this action, the state changes to:

spare_tire_installed
functional_tires
trunk_open
car_lifted
Step 5: Close the Trunk
Finally, we close the trunk.

Action: Close Trunk
Preconditions: trunk_open
Effects: trunk_closed
After this action, the state changes to:

functional_tires
trunk_closed
car_lifted

Final State
After performing all the necessary actions, the final state is:

Goal: functional_tires (all tires are functional)
trunk_closed (trunk is closed)
Thus, the car has a spare tire installed, and the problem is solved.


Key Takeaways:
Planning is a key concept in AI that involves generating a sequence of actions to achieve a goal.
The Spare Tyre Problem is a simple yet effective way to demonstrate AI planning techniques.
STRIPS-like planning involves defining actions with preconditions and effects, which guide the system in choosing actions.
Automated Planning: In more complex environments, such as robotics, AI systems use advanced planning algorithms to handle a variety of tasks, like navigating through obstacles or assembling products.

Applications of Planning in AI:
Robotics: Planning is used to determine the sequence of actions a robot should perform to accomplish tasks such as object manipulation, navigation, and assembly.
Automated Theorem Proving: AI systems use planning algorithms to derive proofs for mathematical statements.
Game AI: Planning helps AI agents in games to plan actions based on the game's state and goals.
Logistics and Scheduling: AI planning is applied to optimize resource allocation and scheduling tasks.
In conclusion, the Spare Tyre Problem provides a simple but effective example of how AI planning works to solve real-world problems using sequence generation, action selection, and goal achievement.
