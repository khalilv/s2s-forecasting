import random
from typing import List, Dict
import json
from collections import deque

def random_trajectories(N: int, targets: List[int], candidates: List[int], M: int = 1000) -> Dict[int, List[List[int]]]:
    """
    Generate up to N random trajectories for each target in targets using numbers from candidates.

    Args:
        N: Maximum number of trajectories to generate per target
        targets: List of target sums to generate trajectories for
        candidates: List of integers to build trajectories from
        M: Maximum number of attempts to generate trajectories per target (default: 1000)

    Returns:
        Dict mapping each target -> list of trajectories summing to that target.
    """
    candidates = sorted(set(candidates))
    output = {}

    for target in targets:
        result = set()
        attempts = 0

        while len(result) < N and attempts < M:
            attempts += 1
            remaining = target
            trajectory: List[int] = []

            while remaining > 0:
                valid = [c for c in candidates if c <= remaining]
                if not valid:
                    break
                choice = random.choice(valid)
                trajectory.append(choice)
                remaining -= choice

            if remaining == 0 and trajectory:
                result.add(tuple(trajectory))

        # convert back to lists, shuffle, and take up to N
        unique = [list(t) for t in result]
        random.shuffle(unique)
        output[target] = unique[:N]

    return output

def build_trajectories(N: int, targets: List[int], candidates: List[int]) -> Dict[int, List[List[int]]]:
    """Build up to N trajectories that sum to each target in targets using numbers from candidates. 
        Each solution will be built from a random previous solution by adding one random candidate.
    
    Args:
        N: Maximum number of trajectories to generate per target
        targets: List of targets to generate trajectories for
        candidates: List of integers to build trajectories from
        
    Returns:
        Dictionary mapping each target to a list of trajectories that sum to it
    """
    candidates = sorted(set(candidates))
    # dp[t] will hold our finalized list of trajectories for sum = t
    dp: Dict[int, List[List[int]]] = {0: [[]]}
    output: Dict[int, List[List[int]]] = {}

    for t in targets:
        found = set()

        # --- 1‑step pass: extend dp[t-c] by one candidate ---
        for c in random.sample(candidates, len(candidates)):
            prev_t = t - c
            if prev_t < 0 or prev_t not in dp:
                continue
            for prefix in random.sample(dp[prev_t], len(dp[prev_t])):
                found.add(tuple(prefix + [c]))
                if len(found) >= N:
                    break
            if len(found) >= N:
                break

        # --- BFS fallback if we still need more ---
        if len(found) < N:
            queue = deque([([], 0)])  # (current_sequence, current_sum)
            seen = set()              # to avoid revisiting the same prefix
            while queue and len(found) < N:
                seq, s = queue.popleft()
                for c in random.sample(candidates, len(candidates)):
                    s2 = s + c
                    if s2 > t:
                        continue
                    new_seq = seq + [c]
                    tup = tuple(new_seq)
                    if tup in seen:
                        continue
                    seen.add(tup)

                    if s2 == t:
                        found.add(tup)
                        if len(found) >= N:
                            break
                    else:
                        queue.append((new_seq, s2))
                # end for c
            # end while queue

        # finalize for this target
        chosen = [list(traj) for traj in found][:N]
        dp[t] = chosen
        output[t] = chosen

    return output

def homogeneous_trajectories(N: int, targets: List[int], candidates: List[int]) -> Dict[int, List[List[int]]]:
    """Build up to N homogeneous trajectories that sum to each target in targets using numbers from candidates.
    A homogeneous trajectory consists of repeated instances of a single candidate value.
    
    Args:
        N: Maximum number of trajectories to generate per target
        targets: List of targets to generate trajectories for 
        candidates: List of integers to build trajectories from
        
    Returns:
        Dictionary mapping each target to a list of homogeneous trajectories that sum to it
    """
    # Sort candidates descending so we prefer the fewest‐step solutions first
    candidates = sorted(set(candidates), reverse=True)

    output: Dict[int, List[List[int]]] = {}
    for t in targets:
        combos: List[List[int]] = []
        for c in candidates:
            if t % c == 0:
                combos.append([c] * (t // c))
            if len(combos) >= N:
                break
        output[t] = combos
    return output

def save_trajectories_to_file(trajectories: Dict[int, List[List[int]]], filename: str) -> None:
    """Save trajectories dictionary to a file.
    
    Args:
        trajectories: Dictionary mapping targets to lists of trajectories
        filename: Path to file to save trajectories to
    """
    with open(filename, 'w') as f:
        json.dump(trajectories, f)

def load_trajectories_from_file(filename: str) -> Dict[int, List[List[int]]]:
    """Load trajectories dictionary from a file.
    
    Args:
        filename: Path to file containing saved trajectories
        
    Returns:
        Dictionary mapping targets to lists of trajectories
    """
    with open(filename, 'r') as f:
        trajectories = json.load(f)
    # Convert string keys back to integers
    return {int(k): v for k, v in trajectories.items()}

def main():
    targets = [6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168]
    candidates = [6,12,18,24,30,36,42,48,54,60,66,72,78,84,90,96,102,108,114,120,126,132,138,144,150,156,162,168]
    N = 100

    print("\nTesting homogeneous_trajectories:")
    homo_traj = homogeneous_trajectories(N, targets, candidates)
    for target, trajectories in homo_traj.items():
        print(f"Target {target}: {trajectories}")

    print("\nTesting random_trajectories:")
    rand_traj = random_trajectories(N, targets, candidates)
    for target, trajectories in rand_traj.items():
        print(f"Target {target}: {trajectories}")

    print("\nTesting build_trajectories:")
    built_traj = build_trajectories(N, targets, candidates)
    for target, trajectories in built_traj.items():
        print(f"Target {target}: {trajectories}")
        print(len(trajectories))

if __name__ == "__main__":
    main()
