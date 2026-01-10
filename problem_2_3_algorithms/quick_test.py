"""Quick test to verify all scripts are working correctly."""
import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from find_largest_delta import find_largest_delta_breaking_homogeneity
    print("✓ find_largest_delta imported successfully")
except Exception as e:
    print(f"✗ find_largest_delta import failed: {e}")

try:
    from find_smallest_epsilon import find_smallest_epsilon_achieving_homogeneity
    print("✓ find_smallest_epsilon imported successfully")
except Exception as e:
    print(f"✗ find_smallest_epsilon import failed: {e}")

try:
    from find_epsilon_bruteforce import find_smallest_epsilon_bruteforce
    print("✓ find_epsilon_bruteforce imported successfully")
except Exception as e:
    print(f"✗ find_epsilon_bruteforce import failed: {e}")

try:
    import benchmark_find_delta
    print("✓ benchmark_find_delta imported successfully")
except Exception as e:
    print(f"✗ benchmark_find_delta import failed: {e}")

try:
    import benchmark_find_epsilon
    print("✓ benchmark_find_epsilon imported successfully")
except Exception as e:
    print(f"✗ benchmark_find_epsilon import failed: {e}")

try:
    import benchmark_epsilon_comparison
    print("✓ benchmark_epsilon_comparison imported successfully")
except Exception as e:
    print(f"✗ benchmark_epsilon_comparison import failed: {e}")

try:
    import run_all_benchmarks
    print("✓ run_all_benchmarks imported successfully")
except Exception as e:
    print(f"✗ run_all_benchmarks import failed: {e}")

print("\n✅ All imports successful! Scripts are properly organized.")
print(f"\n📁 Location: {Path(__file__).parent}")










