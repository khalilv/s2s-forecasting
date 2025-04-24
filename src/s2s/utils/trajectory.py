import json
import random
from typing import List, Dict
from collections import defaultdict

def random_trajectories(N: int, targets: List[int], candidates: List[int], M: int = 10000) -> Dict[int, List[List[int]]]:
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

def build_trajectories(N: int, targets: List[int], candidates: List[int], D: int = 3) -> Dict[int, List[List[int]]]:
    trajectories: Dict[int, List[List[int]]] = defaultdict(list)
    extension_cache: Dict[int, List[List[int]]] = {}

    def generate_valid_extensions(diff: int, max_len: int, max_samples: int = 20) -> List[List[int]]:
        """Generate up to `max_samples` unique combinations of ≤ max_len candidates that sum to `diff`."""
        if max_len == 1:
            return [[c] for c in candidates if c == diff]

        results = []
        seen = set()

        def backtrack(path, total):
            if len(path) > max_len or total > diff or len(results) >= max_samples:
                return
            if total == diff:
                t = tuple(path)
                if t not in seen:
                    seen.add(t)
                    results.append(path[:])
                return
            for c in candidates:
                path.append(c)
                backtrack(path, total + c)
                path.pop()

        try:
            backtrack([], 0)
        except RecursionError:
            return []
        return results

    for i, tgt in enumerate(targets):
        print(tgt)
        seen = set()

        # --- From scratch ---
        direct_valid = generate_valid_extensions(tgt, D, max_samples=5 * N)
        random.shuffle(direct_valid)
        for traj in direct_valid:
            t = tuple(traj)
            if t not in seen:
                seen.add(t)
                trajectories[tgt].append(traj)
                if len(trajectories[tgt]) >= N:
                    break

        if len(trajectories[tgt]) >= N or i == 0:
            continue  # No need for extensions

        # --- Try extensions from earlier targets ---
        attempts = 0
        max_attempts = 100 * N

        while len(trajectories[tgt]) < N and attempts < max_attempts:
            attempts += 1
            prev_idx = random.randint(0, i - 1)
            prev_tgt = targets[prev_idx]
            diff = tgt - prev_tgt

            if diff < 0 or not trajectories[prev_tgt]:
                continue

            base = random.choice(trajectories[prev_tgt])

            if diff not in extension_cache:
                extensions = generate_valid_extensions(diff, D, max_samples=10)
                extension_cache[diff] = extensions  # cache result, even if empty
            else:
                extensions = extension_cache[diff]

            if not extensions:
                continue

            random.shuffle(extensions)
            for ext in extensions:
                new_traj = base + ext
                t = tuple(new_traj)
                if t not in seen:
                    seen.add(t)
                    trajectories[tgt].append(new_traj)
                    break  # Add one valid extension per attempt

        if len(trajectories[tgt]) < N:
            print(f"Warning: Only found {len(trajectories[tgt])} trajectories for target {tgt} (requested {N})")

    return dict(trajectories)

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

def shortest_path_trajectories(N: int, targets: List[int], candidates: List[int]) -> Dict[int, List[List[int]]]:
    candidates = sorted(candidates, reverse=True)  # Prioritize larger steps
    results = {}

    def dfs(path, remaining, all_paths):
        if remaining == 0:
            all_paths.append(path)
            return
        for c in candidates:
            if c <= remaining:
                dfs(path + [c], remaining - c, all_paths)
            if len(all_paths) >= N:
                return

    for target in targets:
        all_paths = []
        dfs([], target, all_paths)
        results[target] = all_paths[:N]

    return results




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
