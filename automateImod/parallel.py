import os
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import List, Optional, Callable, Any, Dict, Tuple, Union

from automateImod.utils import setup_logger

logger = logging.getLogger(__name__)


def chunk_work(items: List[Any], num_chunks: int) -> List[List[Any]]:
    """
    Split a list of items into roughly equal-sized chunks.

    Args:
        items: List of items to split
        num_chunks: Number of chunks to create

    Returns:
        List of chunks (each chunk is a list of items)
    """
    if not items:
        return []

    # Ensure we don't create more chunks than items
    num_chunks = min(num_chunks, len(items))

    # Calculate chunk size
    chunk_size = len(items) // num_chunks
    remainder = len(items) % num_chunks

    chunks = []
    start = 0

    for i in range(num_chunks):
        # Add one extra item to the first 'remainder' chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(items[start:end])
        start = end

    return chunks


def process_batch(func: Callable, batch: List[Any], *args, **kwargs) -> List[Any]:
    """
    Process a batch of items using the provided function.

    Args:
        func: Function to apply to each item
        batch: List of items to process
        *args, **kwargs: Additional arguments to pass to the function

    Returns:
        List of results from processing each item
    """
    results = []
    for item in batch:
        try:
            result = func(item, *args, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {item}: {e}")
            results.append(None)
    return results


def parallel_process(
    items: List[Any],
    process_func: Callable,
    n_workers: Optional[int] = None,
    *args,
    **kwargs,
) -> List[Any]:
    """
    Process items in parallel using a process pool.

    Args:
        items: List of items to process
        process_func: Function to apply to each item
        n_workers: Number of worker processes (default is CPU count)
        *args, **kwargs: Additional arguments to pass to the processing function

    Returns:
        List of results from processing all items
    """
    if not items:
        return []

    # Default to number of CPUs if not specified
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    # Ensure we don't create more workers than items
    n_workers = min(n_workers, len(items))

    # Split items into batches
    batches = chunk_work(items, n_workers)

    # Process batches in parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_results = [
            executor.submit(process_batch, process_func, batch, *args, **kwargs)
            for batch in batches
        ]

        for future in future_results:
            batch_results = future.result()
            results.extend(batch_results)

    return results


def find_tilt_series(data_path: Union[str, Path], pattern: str = "*") -> List[Path]:
    """
    Find all tilt series directories matching a pattern.

    Args:
        data_path: Base directory to search
        pattern: Glob pattern to match

    Returns:
        List of paths to tilt series directories
    """
    data_path = Path(data_path)
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        return []

    # Find all directories matching the pattern
    return list(data_path.glob(pattern))


def parallel_align_tilts(
    data_path: Union[str, Path], basenames: List[str], n_cpu: int = None, **kwargs
) -> List[Dict[str, Any]]:
    """
    Run tilt series alignment in parallel on multiple CPU cores.

    Args:
        data_path: Path to directory containing tilt series data
        basenames: List of tilt series basenames to process
        n_cpu: Number of CPU cores to use (default: all available)
        **kwargs: Additional arguments to pass to the align_tilts function

    Returns:
        List of results from processing each tilt series
    """
    from automateImod.run import process_single_tilt_series

    if not basenames:
        logger.warning(f"No tilt series basenames provided")
        return []

    # Create paths from basenames
    tilt_series_paths = [Path(data_path) / basename for basename in basenames]

    # Process tilt series in parallel
    results = parallel_process(
        tilt_series_paths, process_single_tilt_series, n_workers=n_cpu, **kwargs
    )

    return results
