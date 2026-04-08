#!/usr/bin/env python3
"""
Analyze solving times from receding_horizon_solving_times.csv
"""

import csv
import statistics

def analyze_solving_times(csv_file='receding_horizon_solving_times.csv'):
    """Analyze solving times from CSV file."""
    times = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_sec = float(row['solving_time_seconds'])
            # Only include non-zero times (exclude goal reached iterations)
            if time_sec > 0:
                times.append(time_sec)
    
    if not times:
        print("No valid solving times found!")
        return
    
    # Convert to milliseconds for display
    times_ms = [t * 1000 for t in times]
    
    print("=" * 60)
    print("Solving Time Analysis")
    print("=" * 60)
    print(f"\nTotal iterations: {len(times)}")
    print(f"\nAll iterations (including warm-up):")
    print(f"  Average: {statistics.mean(times_ms):.2f} ms")
    print(f"  Median:  {statistics.median(times_ms):.2f} ms")
    print(f"  Min:      {min(times_ms):.2f} ms")
    print(f"  Max:      {max(times_ms):.2f} ms")
    print(f"  Std Dev:  {statistics.stdev(times_ms):.2f} ms" if len(times) > 1 else "  Std Dev:  N/A")
    print(f"  Total:    {sum(times):.2f} s")
    
    if len(times) > 1:
        # Exclude first iteration (warm-up)
        times_excluding_warmup = times[1:]
        times_ms_excluding_warmup = [t * 1000 for t in times_excluding_warmup]
        
        print(f"\nExcluding first iteration (warm-up):")
        print(f"  Average: {statistics.mean(times_ms_excluding_warmup):.2f} ms")
        print(f"  Median:  {statistics.median(times_ms_excluding_warmup):.2f} ms")
        print(f"  Min:      {min(times_ms_excluding_warmup):.2f} ms")
        print(f"  Max:      {max(times_ms_excluding_warmup):.2f} ms")
        print(f"  Std Dev:  {statistics.stdev(times_ms_excluding_warmup):.2f} ms" if len(times_excluding_warmup) > 1 else "  Std Dev:  N/A")
        print(f"  Total:    {sum(times_excluding_warmup):.2f} s")
    
    print("=" * 60)

if __name__ == '__main__':
    analyze_solving_times()
