import json
import random
from typing import List, Dict, Tuple
from functools import lru_cache

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


def build_trajectories(N: int, targets: List[int], candidates: List[int], D: int) -> Dict[int, List[List[int]]]:
    """
    For each target in `targets`, generate up to N unique trajectories
    (lists of candidates) by extending *any* smaller-target trajectory
    with up to D candidates whose sum exactly equals the difference.
    If fewer than N are found at D, bump D until you either reach N
    or exhaust all possibilities.
    """
    candidates = sorted(candidates)
    min_c, max_c = candidates[0], candidates[-1]

    @lru_cache(maxsize=None)
    def gen_seqs(diff: int, length: int) -> List[Tuple[int, ...]]:
        """All ordered sequences of given length summing to diff,
        pruned by feasibility bounds."""
        out: List[Tuple[int, ...]] = []

        def dfs(rem: int, depth: int, seq: List[int]):
            # prune branches impossible with remaining slots
            if rem < depth * min_c or rem > depth * max_c:
                return
            if depth == 0:
                if rem == 0:
                    out.append(tuple(seq))
                return
            for c in candidates:
                if c > rem:
                    break
                seq.append(c)
                dfs(rem - c, depth - 1, seq)
                seq.pop()

        dfs(diff, length, [])
        return out

    # Start with a “pseudo-target” 0 → one empty trajectory
    result: Dict[int, List[List[int]]] = {0: [[]]}

    for tgt in sorted(targets):
        # collect all smaller-target trajectories
        prevs = [
            (pt, traj)
            for pt, trajs in result.items() if pt < tgt
            for traj in trajs
        ]
        if not prevs:
            prevs = [(0, [])]

        seen = set()
        sols: List[List[int]] = []
        cur_D = D
        max_len = tgt // min_c  # max possible extension length

        # increase D until we gather N or run out
        while len(sols) < N and cur_D <= max_len:
            candidates_pool: List[Tuple[int, ...]] = []
            # build full list of all possible extensions across all bases
            for pt, base_traj in prevs:
                diff = tgt - pt
                # skip impossible diffs
                if diff <= 0:
                    continue
                for length in range(1, cur_D + 1):
                    for ext in gen_seqs(diff, length):
                        candidates_pool.append(tuple(base_traj + list(ext)))

            random.shuffle(candidates_pool)
            # pick unique ones until we hit N
            for traj in candidates_pool:
                if traj not in seen:
                    seen.add(traj)
                    sols.append(list(traj))
                    if len(sols) >= N:
                        break

            if len(sols) < N:
                cur_D += 1

        result[tgt] = sols

    # drop the pseudo-target
    result.pop(0, None)
    return result


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
    # Convert int64 keys to regular integers to avoid JSON serialization issues
    serializable_trajectories = {int(k): v for k, v in trajectories.items()}
    with open(filename, 'w') as f:
        json.dump(serializable_trajectories, f)

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
