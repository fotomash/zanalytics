# Performance Benchmarks

The following table shows a basic timing comparison when fetching three resources using the synchronous `requests` library versus the asynchronous `httpx.AsyncClient`.

| Method      | Time (seconds) |
|-------------|---------------:|
| Synchronous | 0.015 |
| Asynchronous | 0.107 |

The asynchronous approach is generally faster when multiple I/O bound requests are issued concurrently. Actual results will vary based on network conditions and server latency.
